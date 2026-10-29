"""
Faster R-CNN on the KITTI BEV tracks (Complex-YOLO: x, y, w, l, yaw; YOLO3D: + z, h) - 2-Stage reference.

The 1-Stage KITTI rule (backbone + neck with the YOLOv3 layers) would remove the RPN and the RoI head, so the 2-Stage
track keeps Faster R-CNN and adds a rotated-box branch to the RoI head (R2CNN-style, Jiang et al., arXiv 1706.09579):
    RPN and RoI sampling work on the axis-aligned box enclosing each rotated GT box (torchvision Faster R-CNN, unchanged);
    the RoI head predicts, class-agnostically, the rotated box relative to its RoI:
        dx = (x - px) / pw, dy = (y - py) / ph, dw = log(w / pw), dl = log(l / ph), (im, re) = (sin, cos) of the yaw,
        YOLO3D also z (metres, direct) and dh = log(h / pw).
    Loss: smooth L1 (beta 1/9) on the deltas + L1 on (im, re) + the Complex-YOLO unit-norm penalty (1 - im^2 - re^2)^2,
    over the positive RoIs, / #sampled RoIs (the torchvision box-loss normalisation).
Targets and outputs follow the YOLOv3 KITTI layers of this framework (detectors/yolov3): target rows
[b, cls, x, y, w, l, im, re] (complex) / [b, cls, x, y, z, h, w, l, im, re] (yolo3d), x, y, w, l, h normalised by the
image size; output rows [x, y, w, l, im, re, conf, cls...] / [x, y, z, h, w, l, im, re, conf, cls...] in pixels, so the
KITTI evaluation and post-processing of the framework are used unchanged. conf = 1, cls = softmax foreground probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import HeadStats
from detectors.faster_rcnn.build import FasterRCNNDet

NUM_ATTR = {"complex": 6, "yolo3d": 8}          # x y w l im re  /  x y z h w l im re


def rotated_to_aabb(x, y, w, l, im, re):
    """axis-aligned box enclosing the rotated rectangle (half extents w / 2, l / 2 rotated by atan2(im, re))"""
    n = torch.sqrt(im ** 2 + re ** 2).clamp(min=1e-6)
    s, c = (im / n).abs(), (re / n).abs()
    hx = (w * c + l * s) / 2
    hy = (w * s + l * c) / 2
    return torch.stack([x - hx, y - hy, x + hx, y + hy], -1)


class FasterRCNNBEV(FasterRCNNDet):
    def __init__(self, task, num_classes):
        super().__init__(num_classes)
        self.task = task
        rep = self.det.roi_heads.box_head.fc7.out_features
        self.n_rot = NUM_ATTR[task]                                     # deltas + im, re
        self.rot_pred = nn.Linear(rep, self.n_rot)
        nn.init.normal_(self.rot_pred.weight, std=0.001)
        nn.init.zeros_(self.rot_pred.bias)
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _gt(self, targets, B, S):
        """KITTI target rows -> per image dict(boxes AABB px, labels 1..C, rot [n, n_attr] px / metres)"""
        out = []
        for b in range(B):
            t = targets[targets[:, 0] == b]
            if self.task == "complex":
                x, y, w, l = (t[:, 2:6] * S).unbind(1)
                im, re = t[:, 6], t[:, 7]
                rot = torch.stack([x, y, w, l, im, re], 1)
            else:
                x, y = (t[:, 2:4] * S).unbind(1)
                z = t[:, 4]
                h, w, l = (t[:, 5:8] * S).unbind(1)
                im, re = t[:, 8], t[:, 9]
                rot = torch.stack([x, y, z, h, w, l, im, re], 1)
            boxes = rotated_to_aabb(x, y, w, l, im, re).clamp(0, S)
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            out.append({"boxes": boxes[keep], "labels": t[keep, 1].long() + 1, "rot": rot[keep]})
        return out

    def _encode(self, rot, p):
        pw, ph = (p[:, 2] - p[:, 0]).clamp(min=1), (p[:, 3] - p[:, 1]).clamp(min=1)
        px, py = (p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2
        if self.task == "complex":
            x, y, w, l, im, re = rot.unbind(1)
            return torch.stack([(x - px) / pw, (y - py) / ph, torch.log(w / pw), torch.log(l / ph), im, re], 1)
        x, y, z, h, w, l, im, re = rot.unbind(1)
        return torch.stack([(x - px) / pw, (y - py) / ph, z, torch.log(h.clamp(min=1e-3) / pw), torch.log(w / pw),
                            torch.log(l / ph), im, re], 1)

    def _decode(self, d, p):
        pw, ph = (p[:, 2] - p[:, 0]).clamp(min=1), (p[:, 3] - p[:, 1]).clamp(min=1)
        px, py = (p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2
        e = lambda v: torch.exp(v.clamp(max=4.0))  # noqa: E731
        if self.task == "complex":
            return torch.stack([px + d[:, 0] * pw, py + d[:, 1] * ph, e(d[:, 2]) * pw, e(d[:, 3]) * ph, d[:, 4], d[:, 5]], 1)
        return torch.stack([px + d[:, 0] * pw, py + d[:, 1] * ph, d[:, 2], e(d[:, 3]) * pw, e(d[:, 4]) * pw,
                            e(d[:, 5]) * ph, d[:, 6], d[:, 7]], 1)

    def _heads(self, features, proposals, image_sizes):
        rh = self.det.roi_heads
        f = rh.box_head(rh.box_roi_pool(features, proposals, image_sizes))
        class_logits, box_regression = rh.box_predictor(f)
        return class_logits, box_regression, self.rot_pred(f)

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        if targets is None and self.training:                           # PreciseBN (backbone only)
            self._features(x)
            return x.new_zeros(B, 0, self.n_rot + 1 + C)
        if targets is None:
            images, features, _ = self._features(x)
            proposals, _ = self.det.rpn(images, features)
            n = min(len(p) for p in proposals)
            proposals = [p[:n] for p in proposals]
            class_logits, _, rot = self._heads(features, proposals, images.image_sizes)
            dec = self._decode(rot, torch.cat(proposals)).view(B, n, self.n_rot)
            probs = F.softmax(class_logits, -1)[:, 1:].view(B, n, C)
            return torch.cat([dec, torch.ones_like(probs[..., :1]), probs], -1)
        gts = self._gt(targets, B, S)
        tgt = [{"boxes": g["boxes"], "labels": g["labels"]} for g in gts]
        images, features, tgt = self._features(x, tgt)
        proposals, rpn_losses = self.det.rpn(images, features, tgt)
        rh = self.det.roi_heads
        proposals, matched_idxs, labels, regression_targets = rh.select_training_samples(proposals, tgt)
        class_logits, box_regression, rot = self._heads(features, proposals, images.image_sizes)
        from torchvision.models.detection.roi_heads import fastrcnn_loss
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        rot_t = []
        for p, mi, g in zip(proposals, matched_idxs, gts):
            r = g["rot"][mi.clamp(min=0)] if len(g["rot"]) else torch.zeros(len(p), self.n_rot, device=p.device)
            rot_t.append(self._encode(r, p) if len(g["rot"]) else r)
        rot_t, lab = torch.cat(rot_t), torch.cat(labels)
        pos = lab > 0
        n_d = self.n_rot - 2
        if pos.any():
            loss_rot = (F.smooth_l1_loss(rot[pos, :n_d], rot_t[pos, :n_d], beta=1 / 9, reduction="sum")
                        + F.l1_loss(rot[pos, n_d:], rot_t[pos, n_d:], reduction="sum")
                        + ((1 - (rot[pos, n_d:] ** 2).sum(1)) ** 2).sum()) / lab.numel()
        else:
            loss_rot = rot.sum() * 0
        loss_dict = {**rpn_losses, "loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_rot": loss_rot}
        loss = sum(loss_dict.values())
        logit = class_logits.detach()[:, 1:]
        self.stats.update({**loss_dict, "losses": loss, "num_pos": pos.float().sum() / B},
                          F.softmax(class_logits.detach(), -1)[:, 1:].max(-1).values, logit, pos, lab[pos] - 1)
        return loss, x.new_zeros(B, 0, self.n_rot + 1 + C)
