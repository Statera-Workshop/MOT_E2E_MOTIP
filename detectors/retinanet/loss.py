"""
RetinaNet loss: sigmoid focal loss + L1 box regression, normalised by an EMA of the foreground count.

Code : detectron2 modeling/meta_arch/retinanet.py (RetinaNet.losses, RetinaNet.label_anchors),
       dense_detector.py (_ema_update: momentum 0.9, initial 100), box_regression.py
       (_dense_box_regression_loss, "smooth_l1" with Base-RetinaNet.yaml SMOOTH_L1_LOSS_BETA 0.0 = L1),
       fvcore.nn.sigmoid_focal_loss (alpha 0.25, gamma 2, reduction "sum").
Paper: Lin et al. 2017, Eq. 5 (alpha-balanced focal loss) and Sec. 4: "the total focal loss of an image
       is computed as the sum of the focal loss over all ~100k anchors, normalized by the number of
       anchors assigned to a ground-truth box"; ignored anchors (IoU 0.4-0.5) get no loss.
"""

import torch
import torch.nn.functional as F

from detectors.retinanet.box_ops import Box2BoxTransform, Matcher, pairwise_iou


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    """fvcore.nn.focal_loss.sigmoid_focal_loss"""
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def smooth_l1_loss(input, target, beta, reduction="sum"):
    """fvcore.nn.smooth_l1_loss (beta < 1e-5 -> L1)"""
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.sum() if reduction == "sum" else loss.mean()


class Criterion:
    def __init__(self, num_classes, focal_loss_alpha=0.25, focal_loss_gamma=2.0, smooth_l1_beta=0.0):
        self.num_classes = num_classes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box2box_transform = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
        self.anchor_matcher = Matcher((0.4, 0.5), (0, -1, 1), allow_low_quality_matches=True)
        # [Unified] plain attribute as in detectron2: not saved in checkpoints, restarts at 100 per run
        self.loss_normalizer = 100.0

    def _ema_update(self, value, momentum=0.9):
        self.loss_normalizer = self.loss_normalizer * momentum + value * (1 - momentum)
        return self.loss_normalizer

    @torch.no_grad()
    def label_anchors(self, anchors, targets):
        """anchors [R, 4]; targets: list of {"labels", "boxes" XYXY px}. Labels: -1 ignore, K background."""
        gt_labels, matched_gt_boxes = [], []
        for t in targets:
            match_quality_matrix = pairwise_iou(t["boxes"], anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix
            if len(t["labels"]) > 0:
                matched_gt_boxes_i = t["boxes"][matched_idxs]
                gt_labels_i = t["labels"][matched_idxs].clone()
                gt_labels_i[anchor_labels == 0] = self.num_classes
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes
            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def __call__(self, anchors, pred_logits, pred_anchor_deltas, targets):
        """
        anchors            : [R, 4] (all levels concatenated)
        pred_logits        : [B, R, K]
        pred_anchor_deltas : [B, R, 4]
        returns loss dict, pos_mask [B*R], fg labels [num_pos]
        """
        gt_labels, gt_boxes = self.label_anchors(anchors, targets)
        gt_labels = torch.stack(gt_labels)                       # (B, R)
        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        normalizer = self._ema_update(max(num_pos_anchors, 1))

        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1]
        loss_cls = sigmoid_focal_loss(pred_logits[valid_mask], gt_labels_target.to(pred_logits.dtype),
                                      alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum")

        gt_anchor_deltas = torch.stack([self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes])
        loss_box_reg = smooth_l1_loss(pred_anchor_deltas[pos_mask], gt_anchor_deltas[pos_mask],
                                      beta=self.smooth_l1_beta, reduction="sum")

        loss_cls = loss_cls / normalizer
        loss_box_reg = loss_box_reg / normalizer
        losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg, "losses": loss_cls + loss_box_reg,
                  "num_pos": torch.tensor(float(num_pos_anchors) / len(targets))}
        return losses, pos_mask.view(-1), gt_labels[pos_mask]
