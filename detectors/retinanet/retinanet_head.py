"""
RetinaNet classification / box subnets.

Code : detectron2 modeling/meta_arch/retinanet.py RetinaNetHead (NUM_CONVS 4, conv_dims 256, NORM "",
       PRIOR_PROB 0.01); detectron2 _transpose_dense_predictions (dense_detector.py) for the output layout.
Paper: Lin et al. 2017, Sec. 4 and Fig. 3: two FCN subnets of 4 x (3x3 conv 256 + ReLU) shared across
       pyramid levels, last conv K*A (sigmoid) / 4*A; Sec. 4.1 "Initialization": bias of the last
       classification conv = -log((1 - pi) / pi), pi = 0.01.

Init: detectron2 sets every head conv to N(0, 0.01), bias 0, then the prior bias. The framework's
common init (weights_init_normal) later re-draws the conv WEIGHTS from N(0, 0.02) ([Unified]); it does
not touch biases, so the prior bias of cls_score survives (checked in the unit test).
"""

import math

import torch
import torch.nn as nn


class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors, num_convs=4, prior_prob=0.01):
        super().__init__()
        conv_dims = [in_channels] * num_convs
        cls_subnet, bbox_subnet = [], []
        for c_in, c_out in zip([in_channels] + conv_dims, conv_dims):
            cls_subnet += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1), nn.ReLU()]
            bbox_subnet += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1)

        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits, bbox_reg = [], []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


def permute_to_N_HWA_K(tensor, K):
    """detectron2: (N, A*K, Hi, Wi) -> (N, Hi*Wi*A, K)"""
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    return tensor.reshape(N, -1, K)
