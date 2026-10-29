"""
MOT stage (2026-10-23): ground truth and evaluation (HOTA / DetA / AssA, MOTA / IDSW / Frag, IDF1).

GT in memory:  gt[seq] = list over frames of {"ids": [n], "boxes": [n, 4] xyxy, "cls": [n], "ign": [k, 5] xyxy+cls}
               (ign cls -1 = ignore for every class)
Predictions :  pred[seq] = list over frames of [M, 7] = x1, y1, x2, y2, id, class, score

Metrics are TrackEval's own classes (JonathonLuiten/TrackEval: HOTA, CLEAR, Identity) fed with box IoU, per class and
per sequence, combined over sequences with TrackEval's combine_sequences. Reported per class and as the class mean.
[Unified] TrackEval still uses np.float / np.int / np.bool (removed in NumPy 1.24+): the aliases are restored here,
before the import, and nowhere else.
[Unified] ignore regions (KITTI-MOTS id 10000 -> difficult boxes): predictions are first matched to GT (Hungarian on
IoU, >= 0.5, as TrackEval's MOTChallenge preprocessing); an unmatched prediction lying >= 50 % inside an ignore box
of its class is removed. Masks are not used in this box-level evaluation.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

for _a, _t in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402

from common.mot.core import assign, iou_matrix  # noqa: E402

KITTI_CLASSES = ("car", "pedestrian")


def _empty_frame():
    return {"ids": np.zeros(0, int), "boxes": np.zeros((0, 4)), "cls": np.zeros(0, int), "ign": np.zeros((0, 5))}


# ------------------------------------------------------------------ GT loaders
def load_kittimots(voc_root, split="val", classes=KITTI_CLASSES):
    """KITTI_MOTS_VOC (tools/mot/kittimots_to_voc.py) -> gt, frame lists. split: train / val."""
    root = Path(voc_root).expanduser()
    seqmap = (root / "ImageSets" / "MOT" / f"{split}_seqmap.txt").read_text().split("\n")
    gt, frames = {}, {}
    for line in filter(None, seqmap):
        seq, n = line.split()
        seq_frames, ids = [], []
        for f in range(int(n)):
            fid = f"{seq}_{f:06d}"
            fr = _empty_frame()
            objs, ign = [], []
            for o in ET.parse(root / "Annotations" / f"{fid}.xml").getroot().iter("object"):
                bb = o.find("bndbox")
                box = [float(bb.find(k).text) - 1.0 for k in ("xmin", "ymin", "xmax", "ymax")]
                if int(o.find("difficult").text):
                    ign.append(box + [classes.index(o.find("name").text)])
                else:
                    objs.append((int(o.find("track_id").text), box, classes.index(o.find("name").text)))
            if objs:
                fr["ids"] = np.array([o[0] for o in objs])
                fr["boxes"] = np.array([o[1] for o in objs], float)
                fr["cls"] = np.array([o[2] for o in objs])
            if ign:
                fr["ign"] = np.array(ign, float)
            seq_frames.append(fr)
            ids.append(fid)
        gt[seq], frames[seq] = seq_frames, ids
    return gt, frames


def load_fl36mot(root, split="val"):
    """FL36_MOT (tools/mot/make_fl36_mot.py): <root>/<split>/<seq>/gt.txt -> gt, frame image paths."""
    base = Path(root).expanduser() / split
    gt, frames = {}, {}
    for sd in sorted(p for p in base.iterdir() if p.is_dir()):
        n = len(list((sd / "img").glob("*.jpg")))
        seq_frames = [_empty_frame() for _ in range(n)]
        rows = np.loadtxt(sd / "gt.txt", delimiter=",", ndmin=2) if (sd / "gt.txt").stat().st_size else np.zeros((0, 9))
        for f in range(n):
            r = rows[rows[:, 0] == f]
            obj, ign = r[r[:, 7] == 0], r[r[:, 7] == 1]
            seq_frames[f] = {"ids": obj[:, 1].astype(int), "boxes": obj[:, 3:7], "cls": obj[:, 2].astype(int),
                             "ign": np.c_[ign[:, 3:7], ign[:, 2]] if len(ign) else np.zeros((0, 5))}
        gt[sd.name] = seq_frames
        frames[sd.name] = [str(sd / "img" / f"{f:06d}.jpg") for f in range(n)]
    return gt, frames


# ------------------------------------------------------------------ evaluation
def _inside(boxes, ign):
    """fraction of each box inside the ignore boxes (max over ignore boxes)."""
    if len(boxes) == 0 or len(ign) == 0:
        return np.zeros(len(boxes))
    ix1 = np.maximum(boxes[:, None, 0], ign[None, :, 0])
    iy1 = np.maximum(boxes[:, None, 1], ign[None, :, 1])
    ix2 = np.minimum(boxes[:, None, 2], ign[None, :, 2])
    iy2 = np.minimum(boxes[:, None, 3], ign[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    return (inter / area[:, None]).max(1)


def _sequence_data(gt_frames, pred_frames, c):
    gt_ids, tr_ids, sims = [], [], []
    for g, p in zip(gt_frames, pred_frames):
        gm = g["cls"] == c
        gb, gi = g["boxes"][gm], g["ids"][gm]
        p = np.asarray(p).reshape(-1, 7)
        p = p[p[:, 5] == c]
        pb, pi = p[:, :4], p[:, 4].astype(int)
        ign = g["ign"][(g["ign"][:, 4] == c) | (g["ign"][:, 4] < 0)][:, :4] if len(g["ign"]) else np.zeros((0, 4))
        if len(pb) and len(ign):
            sim = iou_matrix(gb, pb)
            matched = np.zeros(len(pb), bool)
            if sim.size:
                m, _, _ = assign(-sim)
                ok = sim[m[:, 0], m[:, 1]] >= 0.5 - np.finfo(float).eps
                matched[m[ok, 1]] = True
            drop = (~matched) & (_inside(pb, ign) >= 0.5)
            pb, pi = pb[~drop], pi[~drop]
        gt_ids.append(gi)
        tr_ids.append(pi)
        sims.append(iou_matrix(gb, pb))
    # contiguous ids 0..N-1 per sequence (TrackEval requirement)
    ug = np.unique(np.concatenate(gt_ids)) if any(len(x) for x in gt_ids) else np.zeros(0, int)
    ut = np.unique(np.concatenate(tr_ids)) if any(len(x) for x in tr_ids) else np.zeros(0, int)
    gmap, tmap = {v: i for i, v in enumerate(ug)}, {v: i for i, v in enumerate(ut)}
    return {"num_timesteps": len(gt_ids), "num_gt_ids": len(ug), "num_tracker_ids": len(ut),
            "num_gt_dets": int(sum(len(x) for x in gt_ids)), "num_tracker_dets": int(sum(len(x) for x in tr_ids)),
            "gt_ids": [np.array([gmap[v] for v in x], int) for x in gt_ids],
            "tracker_ids": [np.array([tmap[v] for v in x], int) for x in tr_ids],
            "similarity_scores": sims}


def evaluate(gt, pred, class_names):
    """-> {"per_class": {name: metrics}, "mean": metrics}; metrics = HOTA DetA AssA MOTA IDF1 IDSW Frag ..."""
    metrics = [HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})]
    per_class = {}
    for c, name in enumerate(class_names):
        if not any((fr["cls"] == c).any() for frames in gt.values() for fr in frames):
            continue                                            # class absent from GT: not part of the mean
        combined = {}
        for M in metrics:
            res = {seq: M.eval_sequence(_sequence_data(gt[seq], pred[seq], c)) for seq in gt}
            combined.update(M.combine_sequences(res))
        per_class[name] = {
            "HOTA": float(np.mean(combined["HOTA"])), "DetA": float(np.mean(combined["DetA"])),
            "AssA": float(np.mean(combined["AssA"])), "LocA": float(np.mean(combined["LocA"])),
            "MOTA": float(combined["MOTA"]), "MOTP": float(combined["MOTP"]), "IDF1": float(combined["IDF1"]),
            "IDSW": int(combined["IDSW"]), "Frag": int(combined["Frag"]),
            "GT_IDs": int(combined["GT_IDs"]) if "GT_IDs" in combined else None,
            "CLR_TP": int(combined["CLR_TP"]), "CLR_FN": int(combined["CLR_FN"]), "CLR_FP": int(combined["CLR_FP"])}
    keys = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1")
    mean = {k: float(np.mean([v[k] for v in per_class.values()])) for k in keys}
    mean.update({k: int(sum(v[k] for v in per_class.values())) for k in ("IDSW", "Frag")})
    return {"per_class": per_class, "mean": mean}
