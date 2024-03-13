# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU



'''
def bbox_overlaps_nwd(bboxes1, bboxes2, eps=1e-7, constant=12.8):

            center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
            center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
            whs = center1[..., :2] - center2[..., :2]

            center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

            w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
            h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
            w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
            h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

            wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

            wassersteins = torch.sqrt(center_distance + wh_distance)

            normalized_wasserstein = torch.exp(-wassersteins/constant)

            return normalized_wasserstein

'''


def bbox_overlaps_nwd(bboxes1, bboxes2, eps=1e-6, C=12.7, xywh=True):
    # Returns Normalized Wasserstein Distance of box1(1,4) to box2(n,4)

    center1 = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
    center2 = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2

    #wh1 = bboxes1[..., 2:] - bboxes1[..., :2]
    #wh2 = bboxes2[..., 2:] - bboxes2[..., :2]

    wh1 = bboxes1[..., 2:]
    wh2 = bboxes2[..., 2:]

    center_distance = ((center1[..., 0] - center2[..., 0]) ** 2 + (center1[..., 1] - center2[..., 1]) ** 2 + eps)

    wh_distance = ((wh1[..., 0] - wh2[..., 0]) ** 2 + (wh1[..., 1] - wh2[..., 1]) ** 2) / 4

    wassersteins = torch.sqrt(center_distance + wh_distance)
    normalized_wasserstein = torch.exp(-wassersteins / C)

    return normalized_wasserstein


def js_divergence_loss(boxes1, boxes2, alpha=0.5):
    """
    Calculate the Jensen-Shannon (JS) divergence between two sets of boxes, using Gaussian distributions.

    Parameters:
    - boxes1 (Tensor): The predicted boxes, shape (m, 4) with [x0, y0, w, h].
    - boxes2 (Tensor): The ground truth boxes, shape (n, 4) with [x0, y0, w, h].
    - alpha (float): The weight for the JS divergence, between 0 and 1.

    Returns:
    - Tensor: The calculated JS divergence for each pair of boxes.
    """

    # Helper function to calculate the covariance matrix for a box
    def covariance_matrix(w, h):
        return torch.tensor([[w ** 2 / 4, 0], [0, h ** 2 / 4]])

    # Initialize JS divergence list
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    js_divergences = []

    # Calculate JS divergence for each pair of boxes
    for box1 in boxes1:
        for box2 in boxes2:
            mu1 = box1[:2]
            sigma1 = covariance_matrix(box1[2], box1[3])

            mu2 = box2[:2]
            sigma2 = covariance_matrix(box2[2], box2[3])

            # Calculate the center of gravity for the mean
            mu_alpha = (1 - alpha) * mu1 + alpha * mu2

            # Calculate the harmonic mean of the covariance matrices
            sigma_alpha_inv = torch.linalg.inv(
                (1 - alpha) * torch.linalg.inv(sigma1) + alpha * torch.linalg.inv(sigma2))

            # Calculate the terms of the JS divergence formula
            tr_term = torch.trace(sigma_alpha_inv @ ((1 - alpha) * sigma1 + alpha * sigma2))
            log_det_term = torch.log(torch.linalg.det(sigma_alpha_inv)) - \
                           ((1 - alpha) * torch.log(torch.linalg.det(sigma1)) +
                            alpha * torch.log(torch.linalg.det(sigma2)))
            mu_diff1 = mu_alpha - mu1
            mu_diff2 = mu_alpha - mu2
            quad_term1 = (1 - alpha) * mu_diff1 @ sigma_alpha_inv @ mu_diff1
            quad_term2 = alpha * mu_diff2 @ sigma_alpha_inv @ mu_diff2

            # Combine the terms to calculate the JS divergence
            js_div = 0.5 * (tr_term + log_det_term - 2 + quad_term1 + quad_term2)
            js_divergences.append(js_div)

    # Convert the list to a tensor
    return torch.tensor(js_divergences)


def js_divergence_loss_vectorized(box1, boxes2, alpha=0.5):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Calculate covariance matrices for the predicted box and ground truth boxes
    Sigma1 = torch.diag_embed(box1[:, 2:] ** 2 / 4)
    Sigma2 = torch.diag_embed(boxes2[:, 2:] ** 2 / 4)

    # Calculate the mean vectors
    mu1 = box1[:, :2]  # Shape (1, 2)
    mu2 = boxes2[:, :2]  # Shape (n, 2)

    # Calculate the center of gravity for the mean
    mu_alpha = (1 - alpha) * mu1 + alpha * mu2  # Shape (n, 2)

    Sigma_alpha = (1 - alpha) * torch.inverse(Sigma1) + alpha * torch.inverse(Sigma2)
    Sigma_alpha_inv = torch.linalg.inv(Sigma_alpha)

    tr_term = torch.diagonal(Sigma_alpha_inv @ ((1 - alpha) * Sigma1 + alpha * Sigma2), dim1=-2, dim2=-1).sum(-1)

    log_det_term = torch.logdet(Sigma_alpha_inv) - ((1 - alpha) * torch.logdet(Sigma1) + alpha * torch.logdet(Sigma2))

    # Compute the quadratic terms for mu
    # Compute the quadratic terms for mu
    mu_diff1 = mu_alpha - mu1  # Shape (n, 2)
    mu_diff2 = mu_alpha - mu2  # Shape (n, 2)
    quad_term1 = (1 - alpha) * mu_diff1 @ Sigma_alpha_inv @ mu_diff1.T
    quad_term2 = alpha * mu_diff2 @ Sigma_alpha_inv @ mu_diff2.T
    quad_term = quad_term1 + quad_term2
    #quad_term = (1 - alpha) * torch.einsum('...i,...ij,...j', mu_diff, Sigma_alpha_inv, mu_diff)

    # Combine the terms to calculate the JS divergence
    js_div = 0.5 * (tr_term + log_det_term - 2 + quad_term)

    # Return a 1D tensor of JS divergences
    return js_div

'''
def bbox_overlaps_nwd(bboxes1, bboxes2, eps=1e-7, C=12.7, xywh=True, weight=2):
    center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    whs = center1[..., :2] - center2[..., :2]

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

    w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
    h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
    w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
    h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / (weight ** 2)

    wassersteins = torch.sqrt(center_distance + wh_distance)

    normalized_wasserstein = torch.exp(-wassersteins / C)

    return normalized_wasserstein
'''

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_nwd(bboxes1, bboxes2, C=12.7, eps=1e-7):
    """
    Return normalized Wasserstein distance between boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        bboxes1 (Tensor[N, 4])
        bboxes2 (Tensor[M, 4])
    Returns:
        nwd (Tensor[N, M]): the NxM matrix containing the pairwise
            Normalized Wasserstein Distance values for every element in bboxes1 and bboxes2
    """

    # Calculate centers (cx, cy) and half widths/heights (half_w, half_h)
    centers1 = (bboxes1[:, 2:] + bboxes1[:, :2]) / 2
    half_sizes1 = (bboxes1[:, 2:] - bboxes1[:, :2]) / 2

    centers2 = (bboxes2[:, 2:] + bboxes2[:, :2]) / 2
    half_sizes2 = (bboxes2[:, 2:] - bboxes2[:, :2]) / 2

    # Compute center distances between each combination of boxes
    center_distance = ((centers1.unsqueeze(1) - centers2.unsqueeze(0)) ** 2).sum(dim=2)

    # Compute size distances between each combination of boxes
    size_distance = ((half_sizes1.unsqueeze(1) - half_sizes2.unsqueeze(0)) ** 2).sum(dim=2)

    # Compute the Wasserstein distance
    wasserstein_distance = torch.sqrt(center_distance + size_distance + eps)

    # Compute the Normalized Wasserstein Distance
    normalized_wasserstein = torch.exp(-wasserstein_distance / C)

    return normalized_wasserstein


def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
