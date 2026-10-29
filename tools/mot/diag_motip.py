#!/usr/bin/env python3
"""
MOT stage (2026-10-25): why does MOTIP associate badly (val HOTA 0.305, AssA 0.22, IDSW 2015)?
Separates "the ID model did not learn enough" from "the runtime tracker (inference port) is wrong".

    UNIFIED_IMAGE_H=384 python tools/mot/diag_motip.py            # ~10 min on the RTX 4070 Laptop

A. teacher-forced ID accuracy, exactly the training computation without augmentation (1 group, no occlusion / switch
   augmentation): consecutive 30-frame clips, DETR outputs matched to GT (Hungarian), matched query outputs as
   trajectory / current features, ID decoder, argmax vs the true ID word. Frames 1..29 (frame 0 has no history).
   Reported for train clips (fit) and val clips (generalisation): accuracy on objects already seen in the clip
   ("known"), on first appearances ("newborn" word), and overall.
B. runtime tracker on the same val clips (consecutive frames, our RuntimeTracker as in the full evaluation),
   evaluated on the clip frames: HOTA / AssA / IDSW, with the official thresholds and with stricter ones.
Reading: A-val known >= ~0.9 but B AssA low -> the inference port is suspect; A-val low -> the ID model is weak
(training budget); A-train high and A-val low -> over-fitting to the 12 train sequences.
Output: stdout + ~/mot_logs/motip_diag.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.datasets.voc import letterbox, to_tensor  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, evaluate, load_kittimots  # noqa: E402
from common.mot.motip import CLIP_LEN, VOCAB, MOTIPNet, id_labels_for_clip, prepare_seq_info, track_sequence  # noqa: E402
from detectors.deformable_detr.build import DeformableDETR  # noqa: E402

ROOT = C.DATASETS_ROOT / "KITTI_MOTS_VOC"


def clip_starts(gt, per_seq):
    out = []
    for s, frames in gt.items():
        n = len(frames) - CLIP_LEN
        if n < 0:
            continue
        for f0 in np.linspace(0, n, per_seq).round().astype(int):
            if sum(len(frames[f]["ids"]) for f in range(f0, f0 + CLIP_LEN)) > 0:
                out.append((s, int(f0)))
    return sorted(set(out))


def load_clip(gt, paths, s, f0, img_size, device):
    imgs, targets, ids = [], [], []
    for f in range(f0, f0 + CLIP_LEN):
        im, sc, px, py = letterbox(Image.open(paths[s][f]).convert("RGB"), img_size)
        g = gt[s][f]
        b = g["boxes"] * sc + np.array([px, py, px, py]) if len(g["boxes"]) else np.zeros((0, 4))
        W, H = im.size
        cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H,
                           (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H], 1).reshape(-1, 4)
        imgs.append(to_tensor(im))
        targets.append({"labels": torch.as_tensor(g["cls"], dtype=torch.long, device=device),
                        "boxes": torch.as_tensor(cxcywh, dtype=torch.float32, device=device)})
        ids.append(g["ids"])
    return torch.stack(imgs).to(device), targets, ids


@torch.no_grad()
def teacher_forced(model, gt, paths, starts, img_size, device):
    crit = model.detr.criterion
    tot = {"known": [0, 0], "newborn": [0, 0], "all": [0, 0]}
    for s, f0 in starts:
        imgs, targets, ids = load_clip(gt, paths, s, f0, img_size, device)
        outs = [model.detr.heads(imgs[k:k + 4]) for k in range(0, CLIP_LEN, 4)]
        logits = torch.cat([o["pred_logits"] for o in outs])
        boxes = torch.cat([o["pred_boxes"] for o in outs])
        hs = [h for o in outs for h in o["hs"]]
        indices = crit.matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)
        info = id_labels_for_clip(ids, np.random.default_rng(0), groups=1, occ_p=0.0, switch_p=0.0)
        if info["traj_labels"].shape[2] == 0:
            continue
        tf, tl, tt, tm, uf, ut, um, ul = prepare_seq_info(hs, indices, info, device)
        pred = model.id_decoder(model.traj(tf), tl, tt, tm, uf, ut, um).argmax(-1)
        valid = ~um
        valid[:, :, 0] = False
        for name, sel in (("known", ul != VOCAB), ("newborn", ul == VOCAB), ("all", torch.ones_like(valid))):
            m = valid & sel
            tot[name][0] += int((pred[m] == ul[m]).sum())
            tot[name][1] += int(m.sum())
    return {k: {"acc": c / n if n else None, "n": n} for k, (c, n) in tot.items()}


def runtime(model, gt, paths, starts, img_size, device, **thr):
    g_clip, p_clip = {}, {}
    for s, f0 in starts:
        key = f"{s}_{f0}"
        g_clip[key] = gt[s][f0:f0 + CLIP_LEN]
        p_clip[key], _ = track_sequence(model, paths[s][f0:f0 + CLIP_LEN], img_size, device, **thr)
    m = evaluate(g_clip, p_clip, KITTI_CLASSES)["mean"]
    return {k: m[k] for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSW")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(C.RUNS_ROOT / "motip" / "motip_kittimots_e005.pth"))
    a = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(a.weights, map_location=device, weights_only=False)
    print("weights:", a.weights, flush=True)
    model = MOTIPNet(DeformableDETR(len(KITTI_CLASSES))).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    res = {}
    for split, per_seq in (("train", 3), ("val", 5)):
        gt, ids = load_kittimots(ROOT, split)
        paths = {s: [str(ROOT / "JPEGImages" / f"{i}.jpg") for i in v] for s, v in ids.items()}
        starts = clip_starts(gt, per_seq)
        res[f"A_{split}"] = teacher_forced(model, gt, paths, starts, ck["img_size"], device)
        print(f"A. teacher-forced ID accuracy, {split} ({len(starts)} clips): "
              + "  ".join(f"{k} {v['acc']:.3f} (n={v['n']})" for k, v in res[f"A_{split}"].items() if v["acc"] is not None),
              flush=True)
        if split == "val":
            for name, thr in (("official det 0.3 / newborn 0.6", {}),
                              ("strict det 0.5 / newborn 0.7", {"det_thresh": 0.5, "newborn_thresh": 0.7})):
                res[f"B_{name}"] = runtime(model, gt, paths, starts, ck["img_size"], device, **thr)
                r = res[f"B_{name}"]
                print(f"B. runtime tracker, val clips, {name}: HOTA {r['HOTA']:.3f} DetA {r['DetA']:.3f} "
                      f"AssA {r['AssA']:.3f} IDF1 {r['IDF1']:.3f} IDSW {r['IDSW']}", flush=True)
    kv = res["A_val"]["known"]["acc"] or 0
    kt = res["A_train"]["known"]["acc"] or 0
    ba = res["B_official det 0.3 / newborn 0.6"]["AssA"]
    if kv >= 0.9 and ba < 0.4:
        verdict = "ID model predicts well under teacher forcing, runtime association is poor -> inference port suspect"
    elif kt >= 0.9 and kv < 0.8:
        verdict = "fits train clips, not val clips -> over-fitting / too little data"
    elif kv < 0.8:
        verdict = "ID model weak even under teacher forcing -> under-trained (training budget)"
    else:
        verdict = "no clear split - look at the numbers"
    res["verdict"] = verdict
    print("VERDICT:", verdict)
    Path(f"~/mot_logs/motip_diag_{Path(a.weights).stem}.json").expanduser().write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
