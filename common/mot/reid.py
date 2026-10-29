"""
MOT stage (2026-10-23): one appearance (ReID) model shared by every tracker that uses appearance (DeepSORT,
BoT-SORT-ReID, later others). [Unified] it is trained from scratch on KITTI-MOTS train track crops (car and
pedestrian together, ids = (sequence, track id)), like the detectors (no pre-training), so all appearance-based
trackers get the same features and differ only in how they use them. The official trackers ship MOT17 pedestrian
models (DeepSORT mars-small128, BoT-SORT FastReID SBS-S50), which do not cover cars and use pre-training.

Model (BoT-style baseline): torchvision ResNet-18 (weights=None), global average pooling -> 512-d feature ->
BNNeck -> linear classifier. Loss = cross entropy (label smoothing 0.1) on the BN feature + batch-hard triplet
(margin 0.3) on the pooled feature. Inference feature = BN feature, L2-normalised (512-d).
Input: crop of the box, letterboxed to 128 x 128 (gray 128 padding), RGB / 255 (no mean / std, as the detectors).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SIZE = 128
MIN_SIDE = 8                     # [Unified] crops with a side below 8 px are not used for training


def crop_tensor(img, box, size=SIZE, jitter=None, rng=None):
    """img: HxWx3 uint8 RGB numpy; box xyxy (pixels) -> 3 x size x size float tensor in [0, 1]."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if jitter and rng is not None:                              # training: small random shift / scale
        bw, bh = x2 - x1, y2 - y1
        dx, dy = rng.uniform(-jitter, jitter, 2) * np.array([bw, bh])
        s = 1.0 + rng.uniform(-jitter, jitter)
        cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
        x1, x2, y1, y2 = cx - bw * s / 2, cx + bw * s / 2, cy - bh * s / 2, cy + bh * s / 2
    x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
    x2, y2 = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return torch.full((3, size, size), 128 / 255.0)
    c = img[y1:y2, x1:x2]
    ch, cw = c.shape[:2]
    sc = size / max(ch, cw)
    nw, nh = max(1, int(round(cw * sc))), max(1, int(round(ch * sc)))
    c = torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float().unsqueeze(0)
    c = F.interpolate(c, size=(nh, nw), mode="bilinear", align_corners=False)[0]
    out = torch.full((3, size, size), 128.0)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[:, oy:oy + nh, ox:ox + nw] = c
    return out / 255.0


class ReIDNet(nn.Module):
    def __init__(self, num_ids):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(r.children())[:-2])
        self.bnneck = nn.BatchNorm1d(512)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, num_ids, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)
        b = self.bnneck(f)
        if self.training:
            return f, self.classifier(b)
        return F.normalize(b, dim=1)


def batch_hard_triplet(f, y, margin=0.3):
    d = torch.cdist(f, f)
    same = y[:, None] == y[None, :]
    hardest_pos = (d * same.float()).max(1).values
    hardest_neg = (d + same.float() * 1e6).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def kitti_crops(voc_root, split):
    """-> list of (image path, box xyxy, global id, class) for every GT object of the split (not ignore regions)."""
    from common.mot.evaluate import load_kittimots
    gt, ids = load_kittimots(voc_root, split)
    out, gid = [], {}
    for seq, frames in gt.items():
        for f, g in enumerate(frames):
            for tid, box, c in zip(g["ids"], g["boxes"], g["cls"]):
                if min(box[2] - box[0], box[3] - box[1]) < MIN_SIDE:
                    continue
                key = (seq, int(tid))
                gid.setdefault(key, len(gid))
                out.append((str(voc_root / "JPEGImages" / f"{ids[seq][f]}.jpg"), box, gid[key], int(c), seq, f))
    return out, gid
