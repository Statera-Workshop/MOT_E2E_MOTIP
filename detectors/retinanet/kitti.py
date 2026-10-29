"""
KITTI track (Complex-YOLO BEV / YOLO3D) for non-YOLO detectors: backbone + neck replaced, YOLOv3 detection layers kept.

Rule of the framework (handoff document): on KITTI only anchor-based detectors are compared, and only their
backbone + neck change; the head is the YOLOv3 Complex-YOLO / YOLO3D layer (detectors/yolov3: anchors, masks,
loss and target assignment from cfg/complex_yolov3.cfg or cfg/yolo3d_yolov3.cfg, unchanged). The neck's
stride-8 / 16 / 32 outputs each get the YOLOv3 output convolution (1x1, linear, 3 x (box attributes + classes)),
exactly what precedes every [yolo] block in the Darknet cfg. Outputs are concatenated in cfg order
(stride 32, 16, 8), like common.darknet.Darknet.forward.

This compares necks on KITTI: FPN (RetinaNet, ResNet-50) and BiFPN (EfficientDet-D0) against Darknet-53 + FPN (v3).
"""

import torch
import torch.nn as nn

from common.darknet import parse_model_config
from detectors.yolov3.build import CFG_DIR, SPECS

YOLO_STRIDES = (32, 16, 8)            # order of the [yolo] blocks in the v3 cfg


class KittiV3Head(nn.Module):
    def __init__(self, task, num_classes, neck_channels, features_fn, precise_bn=True):
        """neck_channels: {stride: channels}; features_fn(x) -> {stride: feature map} for strides 8, 16, 32."""
        super().__init__()
        cfg_name, factory, num_box_attr = SPECS[task]
        blocks = parse_model_config(CFG_DIR / cfg_name)
        hyperparams = dict(blocks[0])
        yolo_blocks = [dict(b) for b in blocks if b["type"] == "yolo"]
        assert len(yolo_blocks) == 3
        for b in yolo_blocks:
            b["classes"] = str(num_classes)
        self.yolo_layers = [factory(b, hyperparams) for b in yolo_blocks]
        self.yolo_modules = nn.ModuleList(self.yolo_layers)
        n_anchor = [len(b["mask"].split(",")) for b in yolo_blocks]
        self.out_convs = nn.ModuleList([
            nn.Conv2d(neck_channels[s], a * (num_box_attr + num_classes), 1, 1, 0, bias=True)
            for s, a in zip(YOLO_STRIDES, n_anchor)])
        self.features_fn = features_fn
        self.precise_bn = precise_bn
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16

    def forward(self, x, targets=None):
        feats = self.features_fn(x)
        img_dim = x.shape[2]
        loss, outs = 0, []
        for s, conv, layer in zip(YOLO_STRIDES, self.out_convs, self.yolo_layers):
            out, layer_loss = layer(conv(feats[s]), targets, img_dim)
            loss = loss + layer_loss
            outs.append(out)
        outs = torch.cat(outs, 1)
        return outs if targets is None else (loss, outs)

    def load_darknet_weights(self, path):
        raise NotImplementedError("no Darknet weights for this backbone; train from scratch.")
