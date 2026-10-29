"""
RetinaNet builder (Image task; KITTI complex / yolo3d track: backbone + neck with the YOLOv3 layers).

Paper: Lin, Goyal, Girshick, He, Dollar, "Focal Loss for Dense Object Detection" (ICCV 2017, arXiv 1708.02002).
Code : detectron2 (facebookresearch, the authors' group; the original Detectron is Caffe2), approved as the
       official-equivalent source on 2026-10-04. Backbone classes from torchvision 0.20.1 (resnet.py).

    backbone  : ResNet-50 (res3, res4, res5)                         resnet.py   (torchvision)
    neck      : FPN P3-P5 + P6/P7 from C5, 256 channels              fpn.py      (detectron2)
    head      : 4 x (3x3 conv + ReLU) cls / box subnets, A = 9        retinanet_head.py
    anchors   : 3 scales x 3 ratios, sizes 32-512 on strides 8-128    box_ops.py
    matching  : IoU >= 0.5 fg, < 0.4 bg, ignore between, low-quality matches allowed
    loss      : focal loss (alpha 0.25, gamma 2) + L1 on deltas / EMA(#fg)   loss.py

Raw contract: [cx, cy, w, h, objectness, class probabilities]. RetinaNet has no objectness branch, so
objectness = 1 (same as v6 / v8 / YOLOX2) and the shared score = sigmoid class probability.

[Unified] differences from detectron2, all from the common protocol:
    no ImageNet weights, trainable BatchNorm, common weights_init_normal (conv weights N(0, 0.02); the
    head's prior bias -log(99) is kept because the common init does not touch biases);
    no pixel mean / std normalisation (same input as every framework detector);
    608 x 608 input, Adam 1e-4, effective batch 8 instead of SGD 0.01, batch 16, 800-pixel shorter side;
    post-processing by the common pipeline (conf 0.01, pre-NMS top-300, NMS 0.5, max 100) instead of
    score > 0.05, top-1000 per level, per-class NMS 0.5.
"""

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.retinanet.box_ops import ANCHOR_SIZES, STRIDES, AnchorGenerator
from detectors.retinanet.fpn import build_retinanet_fpn
from detectors.retinanet.loss import Criterion
from detectors.retinanet.resnet import build_resnet
from detectors.retinanet.retinanet_head import RetinaNetHead, permute_to_N_HWA_K

VARIANT = "retinanet_r50"
RETINANET_CFG = {
    "retinanet_r50":  {"depth": 50,  "fpn_channels": 256, "num_convs": 4, "prior_prob": 0.01},
    "retinanet_r101": {"depth": 101, "fpn_channels": 256, "num_convs": 4, "prior_prob": 0.01},
}


class RetinaNet(nn.Module):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__()
        cfg = RETINANET_CFG[variant]
        self.num_classes = num_classes
        self.strides = STRIDES
        self.backbone = build_resnet(cfg["depth"])
        self.fpn = build_retinanet_fpn(self.backbone.out_channels, cfg["fpn_channels"])
        self.anchor_generator = AnchorGenerator(ANCHOR_SIZES, STRIDES)
        self.head = RetinaNetHead(cfg["fpn_channels"], num_classes, self.anchor_generator.num_cell_anchors,
                                  cfg["num_convs"], cfg["prior_prob"])
        self.criterion = Criterion(num_classes)
        # [Unified] PreciseBN before every evaluation / final save (common/precise_bn.py): with micro batch 2
        # the stored BN statistics gave test mAP50 0.148 vs 0.849 recomputed (FL36, 30 epochs, same weights)
        self.precise_bn = True
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        feats = self.fpn(self.backbone(x))
        logits, bbox_reg = self.head(feats)
        pred_logits = torch.cat([permute_to_N_HWA_K(t, self.num_classes) for t in logits], 1)   # [B, R, K]
        pred_deltas = torch.cat([permute_to_N_HWA_K(t, 4) for t in bbox_reg], 1)                # [B, R, 4]
        anchors = torch.cat(self.anchor_generator([f.shape[-2:] for f in feats], x.device), 0)  # [R, 4]

        B, R = pred_deltas.shape[:2]
        boxes = self.criterion.box2box_transform.apply_deltas(pred_deltas.reshape(-1, 4),
                                                              anchors.repeat(B, 1)).view(B, R, 4)
        raw = to_raw(boxes, pred_logits.sigmoid())          # objectness = 1
        if targets is None:
            return raw

        loss_dict, pos_mask, fg_labels = self.criterion(anchors, pred_logits, pred_deltas,
                                                        split_targets(targets, B, x.shape[2]))
        cls_flat = pred_logits.detach().reshape(-1, self.num_classes)
        self.stats.update(loss_dict, cls_flat.sigmoid().max(-1).values, cls_flat, pos_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("RetinaNet has no Darknet weights; train from scratch.")


class RetinaNetKitti(nn.Module):
    """KITTI track: ResNet-50 + FPN (P3-P5, no P6 / P7) feeding the YOLOv3 Complex-YOLO / YOLO3D layers."""

    def __init__(self, task, num_classes, variant=VARIANT):
        super().__init__()
        from detectors.retinanet.fpn import FPN
        from detectors.retinanet.kitti import KittiV3Head
        cfg = RETINANET_CFG[variant]
        self.backbone = build_resnet(cfg["depth"])
        self.fpn = FPN(self.backbone.out_channels, cfg["fpn_channels"], top_block=None)
        c = cfg["fpn_channels"]
        self.head = KittiV3Head(task, num_classes, {8: c, 16: c, 32: c}, self._features)
        self.yolo_layers = self.head.yolo_layers
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16

    def _features(self, x):
        p3, p4, p5 = self.fpn(self.backbone(x))
        return {8: p3, 16: p4, 32: p5}

    def forward(self, x, targets=None):
        return self.head(x, targets)

    def load_darknet_weights(self, path):
        raise NotImplementedError("RetinaNet has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task == "image":
        return RetinaNet(num_classes)
    if task in ("complex", "yolo3d"):
        return RetinaNetKitti(task, num_classes)
    raise ValueError(f"unknown task {task}")
