"""
Deformable DETR builder (ResNet-50, 4 feature levels, 300 queries, single-stage, no box refinement), Image task.

Paper: Zhu, Su, Lu, Li, Wang & Dai, "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
       (ICLR 2021, arXiv 2010.04159).
Code : official fundamentalvision/Deformable-DETR, files copied unchanged into detectors/deformable_detr/official/
       except the imports: deformable_transformer.py, ms_deform_attn.py, ms_deform_attn_func.py (the compiled
       MultiScaleDeformableAttention CUDA op is optional: without it the official PyTorch reference
       ms_deform_attn_core_pytorch is used - same function, slower), position_encoding.py, matcher.py
       (focal class cost 2, bbox 5, giou 2), deformable_detr.py (SetCriterion with sigmoid focal 0.25 / 2:
       weights ce 2, bbox 5, giou 2, auxiliary losses; MLP), sigmoid_focal_loss of segmentation.py.
       The DeformableDETR module is re-written here like detectors/detr (same computation as the official forward,
       configuration of configs/r50_deformable_detr.sh): res3-res5 -> 1x1 conv + GN(32) per level, 4th level
       3x3 stride-2 conv + GN on res5, sine position embedding per level, 6 + 6 deformable layers (FFN 1024,
       4 points), 300 queries, class prior 0.01, last box layer zero-init with bias[2:] = -2.

Raw contract: 300 queries per image, sigmoid class probabilities (objectness 1); no NMS in the paper ([Unified]).

[Unified] as detectors/detr (no ImageNet weights, trainable BN + PreciseBN, common init of nn.Conv2d / BN - the
input projections' xavier init and the attention Linear init are replaced only where they are Conv2d, no padding
masks, Adam 1e-4 for all parameters instead of AdamW 2e-4 / backbone 2e-5 and gradient clipping 0.1, 30 epochs
instead of 50).
"""

import math

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.deformable_detr.official.deformable_detr import MLP, SetCriterion
from detectors.deformable_detr.official.deformable_transformer import DeformableTransformer
from detectors.deformable_detr.official.matcher import HungarianMatcher
from detectors.deformable_detr.official.position_encoding import PositionEmbeddingSine
from detectors.detr.official.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detectors.detr.official.misc_min import NestedTensor, inverse_sigmoid
from detectors.retinanet.resnet import build_resnet


class DeformableDETR(nn.Module):
    def __init__(self, num_classes, num_queries=300, num_feature_levels=4, hidden_dim=256, aux_loss=True):
        super().__init__()
        self.num_classes, self.num_queries, self.aux_loss = num_classes, num_queries, aux_loss
        self.backbone = build_resnet(50)
        chans = [self.backbone.out_channels[k] for k in ("res3", "res4", "res5")]
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.transformer = DeformableTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6,
                                                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                                                 activation="relu", return_intermediate_dec=True,
                                                 num_feature_levels=num_feature_levels, dec_n_points=4,
                                                 enc_n_points=4, two_stage=False, two_stage_num_proposals=300)
        input_proj_list = [nn.Sequential(nn.Conv2d(c, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim))
                           for c in chans]
        in_channels = chans[-1]
        for _ in range(num_feature_levels - len(chans)):
            input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                                                 nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        class_embed = nn.Linear(hidden_dim, num_classes)
        bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        bias_value = -math.log((1 - 0.01) / 0.01)
        class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = self.transformer.decoder.num_layers
        nn.init.constant_(bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([class_embed for _ in range(num_pred)])   # shared (no box refinement)
        self.bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None
        matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
        for i in range(num_pred - 1):
            weight_dict.update({k + f"_{i}": v for k, v in list(weight_dict.items())[:3]})
        self.criterion = SetCriterion(num_classes, matcher, weight_dict, ["labels", "boxes", "cardinality"],
                                      focal_alpha=0.25)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def heads(self, x):
        feats = self.backbone(x)
        srcs, masks, pos = [], [], []
        for lvl, k in enumerate(("res3", "res4", "res5")):
            f = feats[k]
            m = torch.zeros(f.shape[0], *f.shape[-2:], dtype=torch.bool, device=x.device)
            srcs.append(self.input_proj[lvl](f))
            masks.append(m)
            pos.append(self.position_embedding(NestedTensor(f, m)).to(f.dtype))
        for lvl in range(3, len(self.input_proj)):
            src = self.input_proj[lvl](feats["res5"] if lvl == 3 else srcs[-1])
            m = torch.zeros(src.shape[0], *src.shape[-2:], dtype=torch.bool, device=x.device)
            srcs.append(src)
            masks.append(m)
            pos.append(self.position_embedding(NestedTensor(src, m)).to(src.dtype))
        hs, init_reference, inter_references, _, _ = self.transformer(srcs, masks, pos, self.query_embed.weight)
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            reference = inverse_sigmoid(init_reference if lvl == 0 else inter_references[lvl - 1])
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp.sigmoid())
        out = {"pred_logits": outputs_classes[-1], "pred_boxes": outputs_coords[-1], "hs": hs[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b}
                                  for a, b in zip(outputs_classes[:-1], outputs_coords[:-1])]
        return out

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        WH = x.new_tensor([x.shape[3], x.shape[2], x.shape[3], x.shape[2]])   # [Unified] per-axis (square: = S)
        if targets is None and self.training:                                 # PreciseBN: BN only in the backbone
            self.backbone(x)
            return x.new_zeros(B, 0, 5 + C)
        out = self.heads(x)
        raw = to_raw(box_cxcywh_to_xyxy(out["pred_boxes"]) * WH, out["pred_logits"].sigmoid())
        if targets is None:
            return raw
        tgt = [{"labels": t["labels"], "boxes": box_xyxy_to_cxcywh(t["boxes"] / WH)}
               for t in split_targets(targets, B, S)]
        loss_dict = self.criterion(out, tgt)
        wd = self.criterion.weight_dict
        loss = sum(loss_dict[k] * wd[k] for k in loss_dict if k in wd)
        with torch.no_grad():
            idx = self.criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, tgt)
            fg = torch.zeros(B, self.num_queries, dtype=torch.bool, device=x.device)
            lab = torch.full((B, self.num_queries), -1, dtype=torch.long, device=x.device)
            for b, (i, j) in enumerate(idx):
                fg[b, i] = True
                lab[b, i] = tgt[b]["labels"][j].to(x.device)
            logits = out["pred_logits"].detach().reshape(-1, C)
        self.stats.update({"losses": loss, "loss_ce": loss_dict["loss_ce"], "loss_bbox": loss_dict["loss_bbox"],
                           "loss_giou": loss_dict["loss_giou"], "num_pos": fg.float().sum() / B},
                          logits.sigmoid().max(-1).values, logits, fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("Deformable DETR has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task == "image":
        return DeformableDETR(num_classes)
    if task in ("complex", "yolo3d"):
        from detectors.deformable_detr.kitti import DeformableDETRKitti
        return DeformableDETRKitti(task, num_classes)
    raise ValueError(f"unknown task {task}")
