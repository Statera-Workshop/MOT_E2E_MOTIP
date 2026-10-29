"""
DETR builder (ResNet-50, 6 + 6 transformer layers, 100 queries), Image task.

Paper: Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko, "End-to-End Object Detection with Transformers"
       (ECCV 2020, arXiv 2005.12872).
Code : official facebookresearch/detr, files copied unchanged into detectors/detr/official/ except the imports
       (util.misc -> official/misc_min.py: single process, no padding masks; build() / segmentation unused):
       transformer.py (Transformer, post-norm, return_intermediate_dec), position_encoding.py (PositionEmbeddingSine,
       normalize=True), matcher.py (HungarianMatcher: class 1, bbox 5, giou 2), detr.py (SetCriterion: CE with
       eos_coef 0.1, L1 5, GIoU 2, auxiliary decoding losses; MLP), box_ops.py.
       The DETR module itself is re-written here (40 lines) because the official one takes a NestedTensor and a
       detr backbone object; its computation is the same: res5 -> 1x1 input_proj -> transformer -> class / box heads.
       Backbone: torchvision ResNet-50 classes of detectors/retinanet (res5, stride 32).

Raw contract: the 100 query predictions per image, softmax class probabilities without the no-object class
(objectness 1). DETR uses no NMS; the common post-processing still applies its NMS ([Unified]).

[Unified] no ImageNet weights, trainable BN (FrozenBN in the original) + PreciseBN, common init (conv / BN only;
transformer Linear layers keep the official xavier init), no mean / std, fixed 608 input without padding masks,
Adam 1e-4 for all parameters instead of AdamW 1e-4 / backbone 1e-5 with weight decay 1e-4 and gradient clipping 0.1,
30 epochs (official 500), common augmentation and post-processing.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.detr.official.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detectors.detr.official.detr import MLP, SetCriterion
from detectors.detr.official.matcher import HungarianMatcher
from detectors.detr.official.misc_min import NestedTensor
from detectors.detr.official.position_encoding import PositionEmbeddingSine
from detectors.detr.official.transformer import Transformer
from detectors.retinanet.resnet import build_resnet


class DETR(nn.Module):
    def __init__(self, num_classes, num_queries=100, hidden_dim=256, aux_loss=True, dilation=False):
        super().__init__()
        self.num_classes, self.num_queries, self.aux_loss = num_classes, num_queries, aux_loss
        self.backbone = build_resnet(50, dilate_last=dilation)           # dilation: official DETR-DC5
        self.input_proj = nn.Conv2d(self.backbone.out_channels["res5"], hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.transformer = Transformer(d_model=hidden_dim, dropout=0.1, nhead=8, dim_feedforward=2048,
                                       num_encoder_layers=6, num_decoder_layers=6, normalize_before=False,
                                       return_intermediate_dec=True)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
        for i in range(5):                                                     # dec_layers - 1 auxiliary outputs
            weight_dict.update({k + f"_{i}": v for k, v in list(weight_dict.items())[:3]})
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1,
                                      losses=["labels", "boxes", "cardinality"])
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        # official recipe: gradient clipping 0.1 (main.py --clip_max_norm); applied by common/trainer.py.
        # 300-epoch overfit without it: train-set mAP50 0.0, loss stuck at ~21 (2026-10-08). DETR_CLIP=0 disables.
        self.clip_max_norm = float(os.environ.get("DETR_CLIP", "0.1"))
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def heads(self, x):
        src = self.backbone(x)["res5"]
        mask = torch.zeros(src.shape[0], *src.shape[-2:], dtype=torch.bool, device=x.device)
        pos = self.position_embedding(NestedTensor(src, mask)).to(src.dtype)
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b}
                                  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        WH = x.new_tensor([x.shape[3], x.shape[2], x.shape[3], x.shape[2]])   # [Unified] per-axis (square: = S)
        if targets is None and self.training:                                 # PreciseBN: BN only in the backbone
            self.backbone(x)
            return x.new_zeros(B, 0, 5 + C)
        out = self.heads(x)
        boxes = box_cxcywh_to_xyxy(out["pred_boxes"]) * WH
        raw = to_raw(boxes, F.softmax(out["pred_logits"], -1)[..., :-1])
        if targets is None:
            return raw
        tgt = [{"labels": t["labels"], "boxes": box_xyxy_to_cxcywh(t["boxes"] / WH)}
               for t in split_targets(targets, B, S)]
        loss_dict = self.criterion(out, tgt)
        wd = self.criterion.weight_dict
        loss = sum(loss_dict[k] * wd[k] for k in loss_dict if k in wd)
        with torch.no_grad():                                                   # stats on the last decoder layer
            idx = self.criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, tgt)
            fg = torch.zeros(B, self.num_queries, dtype=torch.bool, device=x.device)
            lab = torch.full((B, self.num_queries), -1, dtype=torch.long, device=x.device)
            for b, (i, j) in enumerate(idx):
                fg[b, i] = True
                lab[b, i] = tgt[b]["labels"][j].to(x.device)
            logits = out["pred_logits"].detach()
            prob = F.softmax(logits, -1)[..., :-1].reshape(-1, C)
        self.stats.update({"losses": loss, "loss_ce": loss_dict["loss_ce"], "loss_bbox": loss_dict["loss_bbox"],
                           "loss_giou": loss_dict["loss_giou"], "num_pos": fg.float().sum() / B},
                          prob.max(-1).values, logits[..., :-1].reshape(-1, C), fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("DETR has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task != "image":
        raise ValueError("DETR: Image task only.")
    return DETR(num_classes)
