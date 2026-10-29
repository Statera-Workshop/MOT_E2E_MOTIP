"""
Shared plumbing for the anchor-free image detectors (YOLOX, YOLOv6, YOLOv8, YOLOX2).

This is the anchor-free counterpart of reusing detectors/yolov3 layers in
v2 / v4 / v5 / v7: everything here is version-independent, so the only
per-version code is backbone / neck / head / label assignment / loss.

    anchor_points()   grid point centres (x + 0.5, y + 0.5) * stride, as in the Legacy code
    decode_exp_boxes() YOLOX / YOLOX2 box parameterisation (t_xy * s + p, exp(t_wh) * s)
    split_targets()   common target tensor [N, 6] -> per-image {"labels", "boxes" (xyxy px)}
    to_raw()          per-point predictions -> common raw contract [B, N, 5 + C]
    HeadStats         object exposing .metrics for common.trainer logging (model.yolo_layers)

Raw contract (tasks/image_task.py, unchanged):
    [cx, cy, w, h (input pixels), objectness, class probabilities]
    Detectors without an objectness branch (v6, v8, YOLOX2) write objectness = 1,
    so the shared score objectness x class = class probability.
"""

import torch


def anchor_points(feat_sizes, strides, device, dtype=torch.float32):
    """List of [H*W, 2] point centres (pixels) and [H*W, 1] strides, one per level."""
    points, stride_t = [], []
    for (h, w), s in zip(feat_sizes, strides):
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                torch.arange(w, device=device, dtype=dtype), indexing="ij")
        xy = torch.stack([xs, ys], -1).view(-1, 2)
        points.append((xy + 0.5) * s)
        stride_t.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return points, stride_t


def split_targets(targets, batch_size, img_size):
    """[N, 6] = [batch, cls, cx, cy, w, h] (normalised) -> list of per-image dicts."""
    out = []
    for b in range(batch_size):
        t = targets[targets[:, 0] == b]
        cxcywh = t[:, 2:6] * img_size
        boxes = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
        out.append({"labels": t[keep, 1].long(), "boxes": boxes[keep]})
    return out


def decode_exp_boxes(reg, points, strides):
    """YOLOX-style box: centre = t_xy * s + p, size = exp(t_wh) * s  ->  xyxy [B, M, 4]."""
    ctr = reg[..., :2] * strides + points
    wh = torch.exp(reg[..., 2:]) * strides
    return torch.cat([ctr - wh * 0.5, ctr + wh * 0.5], -1)


def to_raw(box_xyxy, cls_prob, obj_prob=None):
    """box [B, M, 4] xyxy px, cls_prob [B, M, C], obj_prob [B, M, 1] or None -> [B, M, 5 + C]."""
    cxcy = (box_xyxy[..., :2] + box_xyxy[..., 2:]) / 2
    wh = (box_xyxy[..., 2:] - box_xyxy[..., :2]).clamp(min=0)
    if obj_prob is None:
        obj_prob = torch.ones_like(cls_prob[..., :1])
    return torch.cat([cxcy, wh, obj_prob, cls_prob], -1)


class HeadStats:
    """Stands in for a YOLO layer in model.yolo_layers so common.trainer can log metrics."""

    def __init__(self):
        self.metrics = {}

    @torch.no_grad()
    def update(self, losses, score_prob, cls_logits, fg_mask, fg_labels):
        """
        losses     : dict of scalar tensors (logged as-is)
        score_prob : [B*M] objectness (or max class prob for obj-free heads)
        cls_logits : [B*M, C]
        fg_mask    : [B*M] bool, fg_labels : [num_fg]
        Keys conf_obj / conf_noobj / cls_acc match the anchor-based layers.
        """
        m = {k: float(v.detach()) for k, v in losses.items()}
        if fg_mask.any():
            m["conf_obj"] = float(score_prob[fg_mask].mean())
            m["cls_acc"] = float((cls_logits[fg_mask].argmax(-1) == fg_labels).float().mean() * 100)
        if (~fg_mask).any():
            m["conf_noobj"] = float(score_prob[~fg_mask].mean())
        self.metrics = m


def init_prior_bias(conv, prior_prob=0.01):
    """Focal-loss style prior on classification / objectness logits (Legacy build.py)."""
    b = conv.bias.view(1, -1)
    b.data.fill_(-torch.log(torch.tensor((1.0 - prior_prob) / prior_prob)).item())
    conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
