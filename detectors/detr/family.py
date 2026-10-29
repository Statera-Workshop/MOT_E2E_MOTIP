"""
Shared wrapper for the vendored DETR-family repositories (Conditional DETR, DAB-DETR, DN-DETR, DINO).

Each model is built by the repository's own build function (build / build_DABDETR / build_dino) with the repository's
own default arguments (official/args_official.py, extracted unchanged from main.py and, for DINO, its config files);
only these arguments are overridden ([Framework]): num_classes (dataset), dataset_file, device ("cpu": the runner
moves the module), masks False, and for DN-DETR use_dn True (the DN-DETR method; the repository default runs plain
DAB-DETR). The backbone comes from detectors/detr/adapter.py (ResNet-50, trainable BN, no ImageNet weights).
Training and inference call the model and criterion exactly as the repository's engine.py does.

Raw contract: all queries, sigmoid class probabilities (focal-loss models), boxes from pred_boxes (cx, cy, w, h / image).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.detr.official.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detectors.detr.official.misc_min import NestedTensor


class OfficialDETRFamily(nn.Module):
    def __init__(self, num_classes, build_fn, args_dict, call, overrides=None):
        super().__init__()
        args = SimpleNamespace(**args_dict)
        args.num_classes, args.dataset_file, args.device, args.masks = num_classes, "fl36", "cpu", False
        for k, v in (overrides or {}).items():
            setattr(args, k, v)
        self.args, self.call, self.num_classes = args, call, num_classes
        self.model, self.criterion, _ = build_fn(args)
        self.clip_max_norm = float(getattr(args, "clip_max_norm", 0.1))       # official recipe (common/trainer.py)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _model(self, samples, tgt=None):
        if self.call == "dn":                                                  # DN-DETR engine.py
            a = self.args
            if tgt is not None:
                dn_args = (tgt, a.scalar, a.label_noise_scale, a.box_noise_scale, a.num_patterns)
                if getattr(a, "contrastive", False):
                    dn_args += (a.contrastive,)
                return self.model(samples, dn_args=dn_args)
            return self.model(samples, dn_args=a.num_patterns)[0], None
        if self.call == "targets":                                             # DINO engine.py (need_tgt_for_training)
            return self.model(samples, tgt), None
        return self.model(samples), None                                       # Conditional / DAB-DETR

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        WH = x.new_tensor([x.shape[3], x.shape[2], x.shape[3], x.shape[2]])   # [Unified] per-axis (square: = S)
        samples = NestedTensor(x, torch.zeros(B, S, x.shape[3], dtype=torch.bool, device=x.device))
        if targets is None and self.training:                                  # PreciseBN: BN only in the backbone
            self.model.backbone[0](samples)
            return x.new_zeros(B, 0, 5 + C)
        if targets is None:
            out, _ = self._model(samples)
            return to_raw(box_cxcywh_to_xyxy(out["pred_boxes"]) * WH, out["pred_logits"].sigmoid())
        tgt = [{"labels": t["labels"], "boxes": box_xyxy_to_cxcywh(t["boxes"] / WH)}
               for t in split_targets(targets, B, S)]
        out, mask_dict = self._model(samples, tgt)
        loss_dict = self.criterion(out, tgt, mask_dict) if self.call == "dn" else self.criterion(out, tgt)
        wd = self.criterion.weight_dict
        loss = sum(loss_dict[k] * wd[k] for k in loss_dict if k in wd)
        with torch.no_grad():
            idx = self.criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, tgt)
            fg = torch.zeros(B, out["pred_logits"].shape[1], dtype=torch.bool, device=x.device)
            lab = torch.full_like(fg, -1, dtype=torch.long)
            for b, (i, j) in enumerate(idx):
                fg[b, i] = True
                lab[b, i] = tgt[b]["labels"][j].to(x.device)
            logits = out["pred_logits"].detach().reshape(-1, C)
        self.stats.update({"losses": loss, **{k: loss_dict[k] for k in ("loss_ce", "loss_bbox", "loss_giou") if k in loss_dict},
                           "num_pos": fg.float().sum() / B},
                          logits.sigmoid().max(-1).values, logits, fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, x.new_zeros(B, 0, 5 + C)

    def load_darknet_weights(self, path):
        raise NotImplementedError("no Darknet weights; train from scratch.")
