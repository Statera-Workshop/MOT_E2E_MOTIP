"""
Anchors, anchor matching and box coding for RetinaNet.

Code : detectron2
    modeling/anchor_generator.py  DefaultAnchorGenerator (generate_cell_anchors, _create_grid_offsets,
                                  _grid_anchors); OFFSET 0.0; ASPECT_RATIOS [0.5, 1.0, 2.0] (H/W)
    configs/Base-RetinaNet.yaml   SIZES [[x, x*2^(1/3), x*2^(2/3)] for x in 32, 64, 128, 256, 512]
    modeling/matcher.py           Matcher (IOU_THRESHOLDS [0.4, 0.5], IOU_LABELS [0, -1, 1],
                                  allow_low_quality_matches=True)
    modeling/box_regression.py    Box2BoxTransform (weights 1, 1, 1, 1; scale_clamp log(1000 / 16))
    structures/boxes.py           pairwise_iou
Paper: Lin et al. 2017, Sec. 4 "Anchors": areas 32^2..512^2 on P3..P7, 3 ratios x 3 scales = A = 9;
       IoU >= 0.5 foreground, [0, 0.4) background, [0.4, 0.5) ignored.
"""

import math

import torch

STRIDES = (8, 16, 32, 64, 128)
ANCHOR_SIZES = [[x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]]
ASPECT_RATIOS = (0.5, 1.0, 2.0)
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def generate_cell_anchors(sizes, aspect_ratios=ASPECT_RATIOS):
    """detectron2 DefaultAnchorGenerator.generate_cell_anchors -> [A, 4] XYXY centred at 0."""
    anchors = []
    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
    return torch.tensor(anchors)


class AnchorGenerator:
    """detectron2 DefaultAnchorGenerator, offset 0.0: list of [Hi*Wi*A, 4] XYXY anchors."""

    def __init__(self, sizes=ANCHOR_SIZES, strides=STRIDES, offset=0.0, aspect_ratios=ASPECT_RATIOS):
        self.strides = strides
        self.offset = offset
        self.cell_anchors = [generate_cell_anchors(s, aspect_ratios) for s in sizes]
        self.num_cell_anchors = len(self.cell_anchors[0])

    def __call__(self, feat_sizes, device):
        anchors = []
        for (h, w), stride, base in zip(feat_sizes, self.strides, self.cell_anchors):
            shifts_x = torch.arange(self.offset * stride, w * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(self.offset * stride, h * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base.to(device).view(1, -1, 4)).reshape(-1, 4))
        return anchors


def pairwise_iou(boxes1, boxes2):
    """detectron2 structures/boxes.py pairwise_iou: [M, 4] x [N, 4] XYXY -> [M, N]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    return torch.where(inter > 0, inter / (area1[:, None] + area2 - inter),
                       torch.zeros(1, dtype=inter.dtype, device=inter.device))


class Matcher:
    """detectron2 Matcher: labels -1 ignore, 0 negative, 1 positive."""

    def __init__(self, thresholds=(0.4, 0.5), labels=(0, -1, 1), allow_low_quality_matches=True):
        thresholds = list(thresholds)
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(low <= high for low, high in zip(thresholds[:-1], thresholds[1:]))
        assert all(l in [-1, 0, 1] for l in labels)
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = list(labels)
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            default_match_labels = match_quality_matrix.new_full((match_quality_matrix.size(1),), self.labels[0],
                                                                 dtype=torch.int8)
            return default_matches, default_match_labels
        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)
        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        _, pred_inds_with_highest_quality = (match_quality_matrix == highest_quality_foreach_gt[:, None]).nonzero(
            as_tuple=True)
        match_labels[pred_inds_with_highest_quality] = 1


class Box2BoxTransform:
    """detectron2 Box2BoxTransform (R-CNN deltas dx, dy, dw, dh)."""

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), scale_clamp=_DEFAULT_SCALE_CLAMP):
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights
        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights
        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        return torch.stack((dx, dy, dw, dh), dim=1)

    def apply_deltas(self, deltas, boxes):
        """deltas [R, 4], boxes [R, 4] XYXY -> [R, 4] XYXY"""
        deltas = deltas.float()
        boxes = boxes.to(deltas.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = torch.clamp(deltas[:, 2::4] / ww, max=self.scale_clamp)
        dh = torch.clamp(deltas[:, 3::4] / wh, max=self.scale_clamp)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        return torch.stack((x1, y1, x2, y2), dim=-1).view(deltas.shape)
