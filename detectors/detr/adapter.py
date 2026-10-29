"""
Backbone adapter for the vendored DETR-family code (Conditional DETR, DAB-DETR, DN-DETR, DINO).

The official repositories build `Joiner(backbone, position_embedding)` from an ImageNet torchvision ResNet with
FrozenBatchNorm (models/backbone.py). Here the same interface is built on the framework's ResNet-50
(detectors/retinanet/resnet.py, trainable BN, no pretraining): Joiner is an nn.Sequential whose [0] returns a dict of
NestedTensor feature maps (zero padding masks: fixed 608 x 608 input) and [1] is the repository's own position
encoding; num_channels / strides follow the official BackboneBase (single res5, or res3-res5 for
return_interm_indices [1, 2, 3]).
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from detectors.detr.official.misc_min import NestedTensor
from detectors.retinanet.resnet import build_resnet

LAYER = {1: "res3", 2: "res4", 3: "res5"}


class Body(nn.Module):
    def __init__(self, names, dilation=False):
        super().__init__()
        self.body = build_resnet(50, dilate_last=dilation)       # official --dilation: DC5 (res5 at stride 16)
        self.names = names
        self.num_channels = [self.body.out_channels[n] for n in names]
        self.strides = [{"res3": 8, "res4": 16, "res5": 16 if dilation else 32}[n] for n in names]

    def forward(self, tensor_list):
        feats = self.body(tensor_list.tensors)
        out = OrderedDict()
        for i, n in enumerate(self.names):
            x = feats[n]
            out[str(i)] = NestedTensor(x, torch.zeros(x.shape[0], *x.shape[-2:], dtype=torch.bool, device=x.device))
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels if len(backbone.num_channels) > 1 else backbone.num_channels[0]
        self.strides = backbone.strides

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out, pos = [], []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def make_build_backbone(build_position_encoding):
    def build_backbone(args):
        idx = getattr(args, "return_interm_indices", None)
        names = [LAYER[i] for i in idx] if idx else ["res5"]
        return Joiner(Body(names, bool(getattr(args, "dilation", False))), build_position_encoding(args))
    return build_backbone


class _Registry:
    def register_with_name(self, module_name):
        return lambda f: f

    registe_with_name = register_with_name     # spelling used by the official DINO code


MODULE_BUILD_FUNCS = _Registry()
