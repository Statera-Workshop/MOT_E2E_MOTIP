"""
MOT stage (2026-10-24): MOTIP (Gao, Zhang, Wang, "Multiple Object Tracking as ID Prediction", CVPR 2025), E2E.
Official code MCG-NJU/MOTIP (main, ffc0e90), checked file by file:
  models/motip/id_decoder.py          6 layers; layer 0 cross-attention only, layers 1-5 self-attention among the
                                      current detections then cross-attention to the trajectories; features 256 +
                                      ID embedding 256 (one-hot of 50 ID words + 1 "newborn" word -> linear), heads of
                                      32 dims, add_zero_attn, relative temporal position bias (30 steps, per head),
                                      causal mask (a trajectory token is visible only if its time < current time)
  models/motip/trajectory_modeling.py adapter FFN + LN + FFN + LN on the DETR decoder output (GELU, ratio 2)
  data/transforms.py                  GenerateIDLabels: per clip and per group (6 groups) a random permutation of the
                                      50 ID words; TurnIntoTrajectoryAndUnknown: trajectory occlusion (p 0.5, a random
                                      span masked), trajectory switch (p 0.5 per object per frame, shuffled among the
                                      selected), first appearance -> "newborn" label
  models/motip/id_criterion.py        cross entropy on the current detections, first frame excluded
  train.py                            clip of 30 frames (random interval 1-4), 4 random frames with gradient through
                                      DETR, the rest without; DETR set loss on all frames; the matched query outputs
                                      feed the ID part; AdamW 1e-4 (backbone x0.1, sampling_offsets /
                                      reference_points x0.05), weight decay 5e-4, grad-norm clip 0.1 separately for DETR
                                      and the rest, 10 epochs (x0.1 at 6 and 9), 1 warm-up epoch, batch 1
  models/runtime_tracker.py           det_thresh 0.3, newborn_thresh 0.6, id_thresh 0.2, object-max assignment,
                                      trajectory buffer of the last 30 frames (miss_tolerance 30)

[Unified] decisions:
  * detection part = our KITTI-MOTS Deformable DETR (detectors/deformable_detr, 30 epochs, 1216 x 384, no box
    refinement, 2 classes, RGB / 255) - the same detector as the "deformable_detr" row of the tracking table, so
    MOTIP is compared with tracking-by-detection on identical detection weights (official: COCO + dataset
    pre-trained Deformable DETR with box refinement, ImageNet normalisation).
  * BatchNorm of the detector frozen during MOTIP training (batch 1 clip); the official detector uses
    FrozenBatchNorm2d in the backbone.
  * at most 30 identities per training clip enter the ID part (official NUM_TRAINING_IDS; all objects still train
    the detector), ID decoder layers with activation checkpointing (official USE_DECODER_CHECKPOINT option).
  * reduced schedule for 8 GB / the GPU window: 5 epochs of 1000 random clips (official: every clip start, 10
    epochs), steps x0.1 at epochs 3 and 4, linear warm-up over the first 200 iterations; DETR frames with gradient
    run one at a time with activation checkpointing; common augmentation, identical for all frames of a clip.
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

from common.datasets.voc import hsv_jitter, letterbox, random_scale_translate, to_tensor

VOCAB, GROUPS, CLIP_LEN, REL_PE, MAX_TRAIN_IDS = 50, 6, 30, 30, 30


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, activation):
        super().__init__()
        self.linear1, self.activation, self.linear2 = nn.Linear(d_model, d_ffn), activation, nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class TrajectoryModeling(nn.Module):
    def __init__(self, dim=256, ratio=2):
        super().__init__()
        self.adapter, self.norm = FFN(dim, dim * ratio, nn.GELU()), nn.LayerNorm(dim)
        self.ffn, self.ffn_norm = FFN(dim, dim * ratio, nn.GELU()), nn.LayerNorm(dim)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.norm(x + self.adapter(x))
        return self.ffn_norm(x + self.ffn(x))


class IDDecoder(nn.Module):
    def __init__(self, feature_dim=256, id_dim=256, ratio=2, num_layers=6, head_dim=32, vocab=VOCAB, rel_pe=REL_PE):
        super().__init__()
        self.id_dim, self.num_layers, self.vocab = id_dim, num_layers, vocab
        self.use_checkpoint = True
        d = feature_dim + id_dim
        self.n_heads = d // head_dim
        self.word_to_embed = nn.Linear(vocab + 1, id_dim, bias=False)
        self.embed_to_word = nn.Linear(id_dim, vocab + 1, bias=False)      # shared head (USE_SHARED_AUX_HEAD)
        self.rel_pos_embeds = nn.Parameter(torch.zeros(num_layers, rel_pe, self.n_heads))
        t = torch.arange(rel_pe)
        self.register_buffer("rel_pos_map", t[:, None] - t[None, :], persistent=False)
        mk = lambda: nn.MultiheadAttention(d, self.n_heads, dropout=0.0, batch_first=True, add_zero_attn=True)
        self.self_attn = nn.ModuleList([mk() for _ in range(num_layers - 1)])
        self.self_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers - 1)])
        self.cross_attn = nn.ModuleList([mk() for _ in range(num_layers)])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([FFN(d, d * ratio, nn.GELU()) for _ in range(num_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        for n, p in self.named_parameters():
            if p.dim() > 1 and "rel_pos_embeds" not in n:
                nn.init.xavier_uniform_(p)

    def embed(self, labels):
        return self.word_to_embed(F.one_hot(labels.clamp(min=0), self.vocab + 1).to(self.word_to_embed.weight.dtype))

    def forward(self, traj_feat, traj_labels, traj_times, traj_masks, unk_feat, unk_times, unk_masks):
        """shapes (B, G, T, N[, C]); masks True = padding. -> logits (B, G, Tc, Nc, vocab + 1)"""
        B, G, T, N, _ = traj_feat.shape
        Tc, Nc = unk_feat.shape[2], unk_feat.shape[3]
        traj = torch.cat([traj_feat, self.embed(traj_labels)], -1)
        empty = torch.full(unk_feat.shape[:-1], self.vocab, dtype=torch.long, device=unk_feat.device)
        unk = torch.cat([unk_feat, self.embed(empty)], -1)
        self_kpm = unk_masks.reshape(B * G * Tc, Nc)
        cross_kpm = torch.zeros(B * G, T * N, device=traj.device).masked_fill(traj_masks.reshape(B * G, T * N),
                                                                              float("-inf"))
        tt, ut = traj_times.reshape(B * G, T * N), unk_times.reshape(B * G, Tc * Nc)
        causal = (tt[:, None, :] >= ut[:, :, None]).repeat_interleave(self.n_heads, 0)   # bool (BGH, Lq, Lk)
        rel_idx = self.rel_pos_map[ut[:, :, None].clamp(0, REL_PE - 1), tt[:, None, :].clamp(0, REL_PE - 1)]
        traj_flat = traj.reshape(B * G, T * N, traj.shape[-1])
        for layer in range(self.num_layers):
            if self.training and self.use_checkpoint:                  # official USE_DECODER_CHECKPOINT
                unk = checkpoint(self._layer, layer, unk, traj_flat, self_kpm, cross_kpm, causal, rel_idx,
                                 use_reentrant=False)
            else:
                unk = self._layer(layer, unk, traj_flat, self_kpm, cross_kpm, causal, rel_idx)
        return self.embed_to_word(unk[..., -self.id_dim:])

    def _layer(self, layer, unk, traj_flat, self_kpm, cross_kpm, causal, rel_idx):
        B, G, Tc, Nc, D = unk.shape
        if layer > 0:
            x = unk.reshape(B * G * Tc, Nc, D)
            o, _ = self.self_attn[layer - 1](x, x, x, key_padding_mask=self_kpm)
            unk = self.self_norm[layer - 1](x + o).reshape(B, G, Tc, Nc, D)
        q = unk.reshape(B * G, Tc * Nc, D)
        rel = self.rel_pos_embeds[layer][rel_idx]                                     # (BG, Lq, Lk, H)
        mask = rel.permute(0, 3, 1, 2).reshape(B * G * self.n_heads, Tc * Nc, -1).masked_fill(causal, float("-inf"))
        o, _ = self.cross_attn[layer](q, traj_flat, traj_flat, key_padding_mask=cross_kpm, attn_mask=mask)
        x = self.cross_norm[layer](q + o)
        x = self.ffn_norm[layer](x + self.ffn[layer](x))
        return x.reshape(B, G, Tc, Nc, D)


class MOTIPNet(nn.Module):
    def __init__(self, detr):
        super().__init__()
        self.detr, self.traj, self.id_decoder = detr, TrajectoryModeling(), IDDecoder()

    def freeze_detr_bn(self):
        for m in self.detr.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)


# ------------------------------------------------------------------------------------- clip data and ID labels
def id_labels_for_clip(ids_per_frame, rng, vocab=VOCAB, groups=GROUPS, occ_p=0.5, switch_p=0.5,
                       max_ids=MAX_TRAIN_IDS):
    """GenerateIDLabels + TurnIntoTrajectoryAndUnknown. ids_per_frame: list over T of track-id arrays (annotation
    order). -> dict of (G, T, N) tensors: traj/unk labels, masks (True = absent), ann idxs, times."""
    T = len(ids_per_frame)
    ids = sorted({int(i) for a in ids_per_frame for i in a})
    idx = {v: k for k, v in enumerate(ids)}
    N = len(ids)
    base_mask = np.ones((T, N), bool)
    base_ann = -np.ones((T, N), np.int64)
    for t, a in enumerate(ids_per_frame):
        for i, v in enumerate(a):
            base_mask[t, idx[int(v)]] = False
            base_ann[t, idx[int(v)]] = i
    cap = min(vocab, max_ids)                                         # official NUM_TRAINING_IDS
    if N > cap:
        sel = rng.permutation(N)[:cap]
        base_mask, base_ann, N = base_mask[:, sel], base_ann[:, sel], cap
    labels = np.stack([np.repeat(rng.permutation(vocab)[:N][None], T, 0) for _ in range(groups)])
    t_mask = np.repeat(base_mask[None], groups, 0)
    u_mask = t_mask.copy()
    t_ann = np.repeat(base_ann[None], groups, 0)
    u_ann = t_ann.copy()
    times = np.broadcast_to(np.arange(T)[None, :, None], (groups, T, N)).copy()
    if occ_p > 0:                                                     # trajectory occlusion
        for g in range(groups):
            for n in range(N):
                if random.random() < occ_p:
                    b = random.randint(0, T - 1)
                    e = b + math.ceil((T - 1 - b) * random.random())
                    t_mask[g, b:e, n] = True
                    u_mask[g, b:e, n] = True
    if switch_p > 0:                                                  # trajectory switch
        for g in range(groups):
            for t in range(T):
                sw = np.nonzero(np.random.random(N) < switch_p)[0]
                if len(sw) > 1:
                    sh = sw[np.random.permutation(len(sw))]
                    t_ann[g, t, sw] = t_ann[g, t, sh]
                    t_mask[g, t, sw] = t_mask[g, t, sh]
    u_labels = labels.copy()
    born = np.cumsum(~t_mask, axis=1) > 0                             # (G, T, N)
    newborn = ~np.concatenate([np.zeros((groups, 1, N), bool), born[:, :-1]], 1)
    u_labels[newborn] = vocab
    tt = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    return {"traj_labels": tt(labels), "traj_masks": tt(t_mask), "traj_ann": tt(t_ann), "unk_labels": tt(u_labels),
            "unk_masks": tt(u_mask), "unk_ann": tt(u_ann), "times": tt(times)}


class ClipDataset(torch.utils.data.Dataset):
    """Random 30-frame clips (random interval 1..4, capped by the sequence length) from KITTI-MOTS train."""

    def __init__(self, gt, paths, img_size, clips_per_epoch=1000, aug=None, hflip_prob=0.5, seed=777):
        self.gt, self.paths, self.size, self.n, self.aug, self.hflip = gt, paths, img_size, clips_per_epoch, aug or {}, hflip_prob
        self.starts = [(s, f) for s in gt for f in range(len(gt[s]) - CLIP_LEN + 1)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        s, f0 = self.starts[int(rng.integers(len(self.starts)))]
        remain = len(self.gt[s]) - f0 - 1
        interval = min(int(rng.integers(1, 5)), max(1, remain // (CLIP_LEN - 1)))
        frames = [f0 + interval * k for k in range(CLIP_LEN)]
        geo_seed, flip = int(rng.integers(0, 2 ** 31 - 1)), rng.random() < self.hflip
        imgs, targets, ids = [], [], []
        for f in frames:
            im, sc, px, py = letterbox(Image.open(self.paths[s][f]).convert("RGB"), self.size)
            g = self.gt[s][f]
            boxes = g["boxes"] * sc + np.array([px, py, px, py]) if len(g["boxes"]) else np.zeros((0, 4))
            arr = np.asarray(im).copy()
            st = np.random.get_state()
            np.random.seed(geo_seed)                                  # same colour + geometry for the clip
            arr = hsv_jitter(arr, self.aug.get("hue", 0.1), self.aug.get("saturation", 1.5), self.aug.get("exposure", 1.5))
            arr, boxes, keep = random_scale_translate(arr, boxes, self.aug.get("jitter", 0.2))
            np.random.set_state(st)
            boxes, lab, tid = boxes[keep], g["cls"][keep], g["ids"][keep]   # boxes filtered as labels / ids
            if flip:
                arr = arr[:, ::-1].copy()
                boxes = boxes.copy()
                boxes[:, [0, 2]] = arr.shape[1] - boxes[:, [2, 0]]
            H, W = arr.shape[:2]
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2 / W, (boxes[:, 1] + boxes[:, 3]) / 2 / H
            bw, bh = (boxes[:, 2] - boxes[:, 0]) / W, (boxes[:, 3] - boxes[:, 1]) / H
            imgs.append(to_tensor(arr))
            targets.append({"labels": torch.as_tensor(lab, dtype=torch.long),
                            "boxes": torch.as_tensor(np.stack([cx, cy, bw, bh], 1), dtype=torch.float32).reshape(-1, 4)})
            ids.append(tid)
        info = id_labels_for_clip(ids, rng)
        return torch.stack(imgs), targets, info


def prepare_seq_info(hs_per_frame, indices, info, device):
    """train.py prepare_for_motip for batch 1: matched query outputs placed per group / frame / identity."""
    info = {k: v.to(device) for k, v in info.items()}
    G, T, N = info["traj_labels"].shape
    C = hs_per_frame[0].shape[-1]
    tf = torch.zeros(1, G, T, N, C, device=device)
    uf = torch.zeros(1, G, T, N, C, device=device)
    for t in range(T):
        src, tgt = indices[t][0].to(device), indices[t][1].to(device)
        by_ann = hs_per_frame[t][src[torch.argsort(tgt)]]               # row i = annotation i
        for g in range(G):
            m = ~info["traj_masks"][g, t]
            tf[0, g, t, m] = by_ann[info["traj_ann"][g, t, m]]
            m = ~info["unk_masks"][g, t]
            uf[0, g, t, m] = by_ann[info["unk_ann"][g, t, m]]
    d = lambda k: info[k][None].to(device)
    return tf, d("traj_labels"), d("times"), d("traj_masks"), uf, d("times"), d("unk_masks"), d("unk_labels")


def id_loss(logits, labels, masks):
    """id_criterion.py: first frame excluded, padding removed, CE sum / count."""
    lg, lb, mk = logits[:, :, 1:], labels[:, :, 1:], masks[:, :, 1:]
    lg, lb = lg.reshape(-1, lg.shape[-1])[~mk.reshape(-1)], lb.reshape(-1)[~mk.reshape(-1)]
    return F.cross_entropy(lg, lb, reduction="sum") / max(len(lb), 1)


# ------------------------------------------------------------------------------------- runtime tracker
class RuntimeTracker:
    """models/runtime_tracker.py with assignment "object-max"."""

    def __init__(self, model, det_thresh=0.3, newborn_thresh=0.6, id_thresh=0.2, miss_tolerance=30):
        self.m, self.det_thresh, self.newborn_thresh, self.id_thresh = model, det_thresh, newborn_thresh, id_thresh
        self.miss, self.vocab = miss_tolerance, VOCAB
        self.next_id, self.label_to_id = 0, {}
        self.queue = dict.fromkeys(range(VOCAB))                       # OrderedSet: re-adding moves to the end
        self.tf = self.tl = self.tt = self.tm = None

    def _q_add(self, v):
        self.queue.pop(v, None)
        self.queue[v] = None

    @torch.no_grad()
    def step(self, x):
        out = self.m.detr.heads(x)
        scores, cats = out["pred_logits"][0].sigmoid().max(-1)
        keep = scores > self.det_thresh
        scores, cats, boxes, emb = scores[keep], cats[keep], out["pred_boxes"][0][keep], out["hs"][0][keep]
        dev = x.device
        if self.tf is None or self.tf.shape[0] == 0 or self.tf.shape[1] == 0 or len(boxes) == 0:
            pred = torch.full((len(boxes),), self.vocab, dtype=torch.long, device=dev)
        else:
            T, N = self.tl.shape
            tf = self.m.traj(self.tf)[None, None]
            logits = self.m.id_decoder(tf, self.tl[None, None], self.tt[None, None], self.tm[None, None],
                                       emb[None, None, None], torch.full((1, 1, 1, len(boxes)), T, device=dev, dtype=torch.long),
                                       torch.zeros((1, 1, 1, len(boxes)), dtype=torch.bool, device=dev))[0, 0, 0]
            pred = self._object_max(logits.softmax(-1))
        keep = (pred != self.vocab) | (scores > self.newborn_thresh)
        scores, cats, boxes, emb, pred = scores[keep], cats[keep], boxes[keep], emb[keep], pred[keep]
        n_act = 0
        for p in pred.tolist():
            if p != self.vocab:
                n_act += 1
                self._q_add(p)
        n_new = int((pred == self.vocab).sum())
        remaining = len(self.queue) - n_act
        if n_new > remaining:
            kn = torch.ones(len(pred), dtype=torch.bool, device=dev)
            nb = torch.nonzero(pred == self.vocab)[:, 0]
            kn[nb[remaining:]] = False
            scores, cats, boxes, emb, pred = scores[kn], cats[kn], boxes[kn], emb[kn], pred[kn]
        labels = self._assign_newborn(pred)
        res = (boxes, labels, scores, cats)
        for v in labels.tolist():
            self._q_add(v)
        self._update(boxes, emb, labels)
        return [self.label_to_id[v] for v in labels.tolist()], res

    def _object_max(self, sc):
        tracked = set(self.tl[0].tolist())
        conf, lab = sc.max(-1)
        best = {}
        for c, l in zip(conf.tolist(), lab.tolist()):
            best[l] = max(best.get(l, -1.0), c)
        if self.vocab in best:
            best[self.vocab] = 0.0
        out = []
        for c, l in zip(conf.tolist(), lab.tolist()):
            out.append(self.vocab if (l not in tracked or c < self.id_thresh or c < best[l]) else l)
        return torch.tensor(out, dtype=torch.long, device=sc.device)

    def _assign_newborn(self, pred):
        nb = int((pred == self.vocab).sum())
        if nb == 0:
            return pred
        new = list(self.queue)[:nb]
        if self.tl is not None and self.tl.shape[0] > 0:
            rm = torch.zeros(self.tl.shape[1], dtype=torch.bool, device=pred.device)
            for v in new:
                rm |= self.tl[0] == v
            self.tf, self.tl, self.tt, self.tm = self.tf[:, ~rm], self.tl[:, ~rm], self.tt[:, ~rm], self.tm[:, ~rm]
        for v in new:
            self.label_to_id.pop(v, None)
        pred = pred.clone()
        pred[pred == self.vocab] = torch.tensor(new, dtype=torch.long, device=pred.device)
        for v in new:
            self.label_to_id[v] = self.next_id
            self.next_id += 1
        return pred

    def _update(self, boxes, emb, labels):
        dev, C = emb.device, 256
        if self.tl is None:
            self.tf = torch.zeros(0, 0, C, device=dev)
            self.tl = torch.zeros(0, 0, dtype=torch.long, device=dev)
            self.tt = torch.zeros(0, 0, dtype=torch.long, device=dev)
            self.tm = torch.zeros(0, 0, dtype=torch.bool, device=dev)
        k = -self.miss + 2
        self.tf, self.tl, self.tt, self.tm = self.tf[k:], self.tl[k:], self.tt[k:], self.tm[k:]
        have = set(self.tl[0].tolist()) if self.tl.shape[0] > 0 else set()
        new = sorted(set(labels.tolist()) - have)
        T = self.tl.shape[0]
        if new:
            n = torch.tensor(new, dtype=torch.long, device=dev)
            self.tl = torch.cat([self.tl, n[None].repeat(T, 1)], 1)
            self.tf = torch.cat([self.tf, torch.zeros(T, len(new), C, device=dev)], 1)
            self.tt = torch.cat([self.tt, torch.arange(T, device=dev)[:, None].repeat(1, len(new))], 1)
            self.tm = torch.cat([self.tm, torch.ones(T, len(new), dtype=torch.bool, device=dev)], 1)
        N = self.tl.shape[1]
        cur_l = self.tl[0].clone() if T > 0 else labels.clone()
        cur_f = torch.zeros(N, C, device=dev)
        cur_m = torch.ones(N, dtype=torch.bool, device=dev)
        ij = torch.eq(cur_l[:, None], labels[None, :]).nonzero()
        cur_l[ij[:, 0]] = labels[ij[:, 1]]
        cur_f[ij[:, 0]] = emb[ij[:, 1]]
        cur_m[ij[:, 0]] = False
        self.tf = torch.cat([self.tf, cur_f[None]], 0)
        self.tl = torch.cat([self.tl, cur_l[None]], 0)
        self.tm = torch.cat([self.tm, cur_m[None]], 0)
        self.tt = torch.arange(self.tl.shape[0], device=dev)[:, None].repeat(1, self.tl.shape[1])
        act = (~self.tm).sum(0) > 0
        self.tf, self.tl, self.tt, self.tm = self.tf[:, act], self.tl[:, act], self.tt[:, act], self.tm[:, act]


@torch.no_grad()
def track_sequence(model, frame_paths, img_size, device, upto=None, **thr):
    trk = RuntimeTracker(model, **thr)
    preds, times = [], []
    for p in (frame_paths[:upto] if upto else frame_paths):
        im = Image.open(p).convert("RGB")
        lb, sc, px, py = letterbox(im, img_size)
        x = to_tensor(lb).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids, (boxes, _, scores, cats) = trk.step(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        H, W = x.shape[2:]
        b = boxes.cpu().numpy()
        o = np.zeros((len(b), 7))
        if len(b):
            x1, y1 = (b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H
            x2, y2 = (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H
            o[:, 0], o[:, 2] = (x1 - px) / sc, (x2 - px) / sc
            o[:, 1], o[:, 3] = (y1 - py) / sc, (y2 - py) / sc
            o[:, 4], o[:, 5], o[:, 6] = ids, cats.cpu().numpy(), scores.cpu().numpy()
        preds.append(o)
    return preds, times
