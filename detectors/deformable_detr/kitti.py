"""
Deformable DETR on the KITTI BEV tracks (DETR-family KITTI reference, 2026-10-08).

As for the 2-Stage Faster R-CNN BEV track, the YOLOv3 KITTI layers cannot be reused (no anchors / set prediction), so
the query decoder keeps its own head and gets a rotated-box branch:
    Hungarian matching and the official box / class losses work on the axis-aligned box enclosing each rotated GT
    (detectors/faster_rcnn/kitti_bev.rotated_to_aabb); a 3-layer MLP on the last decoder state predicts, per query,
    w, l (sigmoid, / image), (im, re) and for YOLO3D also z (metres) and h (sigmoid, / image); the centre is the
    centre of the predicted axis-aligned box. Loss on the matched queries of the last layer: 5 x L1 (w, l, z, h) +
    L1 (im, re) + unit-norm penalty (1 - im^2 - re^2)^2, / #GT (the official box-loss normalisation).
Outputs follow the Complex-YOLO / YOLO3D row layout of detectors/yolov3 (conf 1, cls = sigmoid class probabilities),
so the framework's KITTI post-processing and evaluation are used unchanged. KITTI lr: Adam 1e-4 (as all DETR-family).
"""

import torch
import torch.nn.functional as F

from detectors.deformable_detr.build import DeformableDETR
from detectors.deformable_detr.official.deformable_detr import MLP
from detectors.detr.official.box_ops import box_xyxy_to_cxcywh
from detectors.faster_rcnn.kitti_bev import NUM_ATTR, rotated_to_aabb


class DeformableDETRKitti(DeformableDETR):
    def __init__(self, task, num_classes):
        super().__init__(num_classes)
        self.task = task
        self.n_rot = NUM_ATTR[task]
        self.rot_embed = MLP(256, 256, 4 if task == "complex" else 6, 3)        # w l im re (+ z h)
        self.clip_max_norm = 0.1

    def _gt(self, targets, B, S):
        out = []
        for b in range(B):
            t = targets[targets[:, 0] == b]
            if self.task == "complex":
                x, y, w, l = (t[:, 2:6] * S).unbind(1)
                im, re = t[:, 6], t[:, 7]
                extra = torch.stack([w / S, l / S, im, re], 1)
            else:
                x, y = (t[:, 2:4] * S).unbind(1)
                z = t[:, 4]
                h, w, l = (t[:, 5:8] * S).unbind(1)
                im, re = t[:, 8], t[:, 9]
                extra = torch.stack([w / S, l / S, im, re, z, h / S], 1)
            boxes = rotated_to_aabb(x, y, w, l, im, re).clamp(0, S)
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            out.append({"labels": t[keep, 1].long(), "boxes": box_xyxy_to_cxcywh(boxes[keep] / S), "rot": extra[keep]})
        return out

    def _rot(self, hs):
        r = self.rot_embed(hs)
        if self.task == "complex":
            return torch.cat([r[..., :2].sigmoid(), r[..., 2:4]], -1)
        return torch.cat([r[..., :2].sigmoid(), r[..., 2:4], r[..., 4:5], r[..., 5:6].sigmoid()], -1)

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        if targets is None and self.training:
            self.backbone(x)
            return x.new_zeros(B, 0, self.n_rot + 1 + C)
        out = self.heads(x)
        rot = self._rot(out.pop("hs"))
        if targets is None:
            cxcy = out["pred_boxes"][..., :2] * S
            wl, ang = rot[..., :2] * S, rot[..., 2:4]
            if self.task == "complex":
                rows = torch.cat([cxcy, wl, ang], -1)
            else:
                rows = torch.cat([cxcy, rot[..., 4:5], rot[..., 5:6] * S, wl, ang], -1)
            probs = out["pred_logits"].sigmoid()
            return torch.cat([rows, torch.ones_like(probs[..., :1]), probs], -1)
        gts = self._gt(targets, B, S)
        tgt = [{"labels": g["labels"], "boxes": g["boxes"]} for g in gts]
        loss_dict = self.criterion(out, tgt)
        wd = self.criterion.weight_dict
        loss = sum(loss_dict[k] * wd[k] for k in loss_dict if k in wd)
        idx = self.criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, tgt)
        src = torch.cat([rot[b, i] for b, (i, _) in enumerate(idx)])
        dst = torch.cat([gts[b]["rot"][j.to(x.device)] for b, (_, j) in enumerate(idx)])
        n = max(float(len(dst)), 1.0)
        if len(dst):
            size_cols = [0, 1] if self.task == "complex" else [0, 1, 4, 5]
            loss_rot = (5 * F.l1_loss(src[:, size_cols], dst[:, size_cols], reduction="sum")
                        + F.l1_loss(src[:, 2:4], dst[:, 2:4], reduction="sum")
                        + ((1 - (src[:, 2:4] ** 2).sum(1)) ** 2).sum()) / n
        else:
            loss_rot = rot.sum() * 0
        loss = loss + loss_rot
        with torch.no_grad():
            fg = torch.zeros(B, self.num_queries, dtype=torch.bool, device=x.device)
            lab = torch.full((B, self.num_queries), -1, dtype=torch.long, device=x.device)
            for b, (i, j) in enumerate(idx):
                fg[b, i] = True
                lab[b, i] = tgt[b]["labels"][j].to(x.device)
            logits = out["pred_logits"].detach().reshape(-1, C)
        self.stats.update({"losses": loss, "loss_ce": loss_dict["loss_ce"], "loss_bbox": loss_dict["loss_bbox"],
                           "loss_rot": loss_rot, "num_pos": fg.float().sum() / B},
                          logits.sigmoid().max(-1).values, logits, fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, x.new_zeros(B, 0, self.n_rot + 1 + C)


