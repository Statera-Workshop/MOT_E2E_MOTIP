"""
Faster R-CNN (ResNet-50-FPN) builder - the 2-Stage reference. Image task.

Paper: Ren, He, Girshick & Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
       (NeurIPS 2015, arXiv 1506.01497); FPN: Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017).
Code : torchvision 0.20.1 models/detection (fasterrcnn_resnet50_fpn: resnet_fpn_backbone, AnchorGenerator,
       RPNHead / RegionProposalNetwork, MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor, RoIHeads), used directly;
       torchvision is the approved official-equivalent source (the authors' code is Caffe / Detectron).
       The training path calls RoIHeads.select_training_samples + fastrcnn_loss exactly like RoIHeads.forward, so the
       sampled RoI logits can also be logged.

    backbone  ResNet-50, all stages trainable, BatchNorm2d (torchvision default FrozenBatchNorm2d needs ImageNet stats)
    neck      FPN P2-P5 + P6 (max-pool)                       RPN  anchors 32-512 x (0.5, 1, 2), IoU 0.7 / 0.3, 256 / img
    RoI head  MultiScaleRoIAlign 7x7 (P2-P5) -> 2 x FC 1024 -> class (C + 1) + class-specific box (10, 10, 5, 5)
              512 RoIs / img, 25 % positives, IoU 0.5

Raw contract: one row per (proposal, class): the class-specific box and a class vector holding only that class's
softmax probability (objectness 1). The common max-class + NMS post-processing on these rows is torchvision's
per-class post-processing (score > thr, per-class NMS).

[Unified] no ImageNet weights, trainable BN (FrozenBN in the original), common init, transform without mean / std and
without resizing (608 input), Adam 1e-4, common post-processing (instead of score 0.05, NMS 0.5, top-100),
PreciseBN (BN ResNet-50 + FPN, micro batch 2: same reason as RetinaNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import fastrcnn_loss

from detectors.anchor_free.common import HeadStats, split_targets


class FasterRCNNDet(nn.Module):
    def __init__(self, num_classes, img_size=608):
        super().__init__()
        self.num_classes = num_classes
        backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, norm_layer=nn.BatchNorm2d,
                                       trainable_layers=5)
        self.det = FasterRCNN(backbone, num_classes=num_classes + 1, min_size=img_size, max_size=img_size,
                              image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0])
        self.det.transform.resize = lambda image, target=None: (image, target)     # [Unified] no resizing at all
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def neck(self, features):
        """hook: extra neck after the FPN (Libra R-CNN: balanced feature pyramid)"""
        return features

    def _features(self, x, tgt=None):
        images, tgt = self.det.transform(list(x), tgt)
        return images, self.neck(self.det.backbone(images.tensors)), tgt

    def sample(self, proposals, tgt):
        """hook: RoI sampling -> proposals, labels, regression targets (torchvision RoIHeads.select_training_samples)"""
        proposals, _, labels, regression_targets = self.det.roi_heads.select_training_samples(proposals, tgt)
        return proposals, labels, regression_targets

    def box_loss(self, class_logits, box_regression, labels, regression_targets):
        """hook: second-stage loss (torchvision fastrcnn_loss: CE + smooth L1 beta 1/9, / #samples)"""
        return fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

    def roi_outputs(self, features, proposals, image_sizes):
        rh = self.det.roi_heads
        box_features = rh.box_head(rh.box_roi_pool(features, proposals, image_sizes))
        return rh.box_predictor(box_features)

    def to_raw(self, class_logits, box_regression, proposals, B):
        """per (proposal, class) rows -> [B, P * C, 5 + C] (P proposals per image)"""
        C = self.num_classes
        boxes = self.det.roi_heads.box_coder.decode(box_regression, proposals)[:, 1:]          # [sum P, C, 4]
        scores = F.softmax(class_logits, -1)[:, 1:]                                              # [sum P, C]
        P = boxes.shape[0] // B
        boxes, scores = boxes.reshape(B, P * C, 4), scores.reshape(B, P * C)
        cls = torch.zeros(B, P * C, C, device=scores.device, dtype=scores.dtype)
        cls.view(B, P, C, C).diagonal(dim1=2, dim2=3).copy_(scores.view(B, P, C))
        cxcy = (boxes[..., :2] + boxes[..., 2:]) / 2
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
        return torch.cat([cxcy, wh, torch.ones_like(scores[..., None]), cls], -1)

    def forward(self, x, targets=None):
        B, C = x.shape[0], self.num_classes
        if targets is None and self.training:
            # PreciseBN (common/precise_bn.py) runs train-mode forwards without targets to re-estimate BN statistics;
            # all BN layers are in the backbone, and torchvision's RPN refuses train mode without targets.
            self._features(x)
            return x.new_zeros(B, 0, 5 + C)
        if targets is None:
            images, features, _ = self._features(x)
            proposals, _ = self.det.rpn(images, features)
            n = min(len(p) for p in proposals)                                                   # equal count per image
            proposals = [p[:n] for p in proposals]
            class_logits, box_regression = self.roi_outputs(features, proposals, images.image_sizes)
            return self.to_raw(class_logits, box_regression, proposals, B)
        tgt = [{"boxes": t["boxes"], "labels": t["labels"] + 1} for t in split_targets(targets, B, x.shape[2])]
        images, features, tgt = self._features(x, tgt)
        proposals, rpn_losses = self.det.rpn(images, features, tgt)
        proposals, labels, regression_targets = self.sample(proposals, tgt)
        class_logits, box_regression = self.roi_outputs(features, proposals, images.image_sizes)
        loss_classifier, loss_box_reg = self.box_loss(class_logits, box_regression, labels, regression_targets)
        loss_dict = {**rpn_losses, "loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        loss = sum(loss_dict.values())
        lab = torch.cat(labels)
        fg = lab > 0
        logit = class_logits.detach()[:, 1:]
        self.stats.update({**loss_dict, "losses": loss, "num_pos": fg.float().sum() / B},
                          F.softmax(class_logits.detach(), -1)[:, 1:].max(-1).values, logit, fg, lab[fg] - 1)
        return loss, x.new_zeros(B, 0, 5 + C)

    def load_darknet_weights(self, path):
        raise NotImplementedError("Faster R-CNN has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task == "image":
        return FasterRCNNDet(num_classes)
    if task in ("complex", "yolo3d"):                                   # 2-Stage KITTI BEV track
        from detectors.faster_rcnn.kitti_bev import FasterRCNNBEV
        return FasterRCNNBEV(task, num_classes)
    raise ValueError(f"unknown task {task}")
