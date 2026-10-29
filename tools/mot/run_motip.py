#!/usr/bin/env python3
"""
MOT stage (2026-10-24): run a trained MOTIP online on KITTI-MOTS val and evaluate (same evaluation as
run_tracker.py). Latency = whole pipeline per frame (DETR forward, ID decoder over the trajectory buffer, ID assignment), batch 1, fp32, cuda synchronize, after 20 warm-up frames; letterbox not timed.

    UNIFIED_IMAGE_H=384 python tools/mot/run_motip.py [--weights ...] [--max-seqs 1] [--causal]

--causal: two sequences, two prefixes each; the last frame of a prefix run must reproduce the full run (same ids,
boxes within 1e-3 px - GPU kernels may differ in the last bits).
Output: ~/mot_logs/track/kittimots_motip/joint.json (+ summary.txt), as a "joint" method row in mot_table.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.motip import MOTIPNet, track_sequence  # noqa: E402
from detectors.deformable_detr.build import DeformableDETR  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, evaluate, load_kittimots  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(C.RUNS_ROOT / "motip" / "motip_kittimots_e005.pth"))
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--name", default="motip")
    a = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(Path(a.weights).expanduser(), map_location=device, weights_only=False)
    model = MOTIPNet(DeformableDETR(len(KITTI_CLASSES))).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
    gt, ids = load_kittimots(root, "val")
    if a.max_seqs:
        gt = dict(list(gt.items())[:a.max_seqs])
    paths = {s: [str(root / "JPEGImages" / f"{i}.jpg") for i in ids[s]] for s in gt}
    first = next(iter(paths.values()))
    track_sequence(model, first[:20], ck["img_size"], device)       # warm-up
    pred, times = {}, []
    for s in gt:
        pred[s], t = track_sequence(model, paths[s], ck["img_size"], device)
        times += t
        print(f"  {s}: {len(t)} frames, {np.median(t):.1f} ms median", flush=True)
    causal = None
    if a.causal:
        causal = "OK"
        rng = np.random.default_rng(777)
        for s in list(gt)[:2]:
            for upto in sorted(rng.integers(2, len(paths[s]) + 1, 2)):
                p, _ = track_sequence(model, paths[s], ck["img_size"], device, upto=upto)
                x, y = p[-1], pred[s][upto - 1]
                if x.shape != y.shape or not np.array_equal(x[:, 4], y[:, 4]) or not np.allclose(x, y, atol=1e-3):
                    causal = f"FAIL ({s} frame {upto - 1})"
    res = evaluate(gt, pred, KITTI_CLASSES)
    tm = np.array(times)
    rec = {"tracker": "joint", "method": a.name, "family": "E2E", "run": a.name, "source": "kittimots",
           "split": "val", "weights": str(a.weights), "thresholds": "det 0.3, newborn 0.6, id 0.2, object-max, miss 30",
           "detector_ms": {"median": float(np.median(tm)), "p95": float(np.percentile(tm, 95))},
           "tracker_ms": {"median": 0.0, "p95": 0.0, "note": "included in detector_ms (one network)"},
           "reid_ms": None, "score_thr": None, "causal_check": causal, **res}
    out = Path("~/mot_logs/track").expanduser() / f"kittimots_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "joint.json").write_text(json.dumps(rec, indent=1))
    m = res["mean"]
    line = (f"{a.name:<12} HOTA {m['HOTA']:.4f}  DetA {m['DetA']:.4f}  AssA {m['AssA']:.4f}  MOTA {m['MOTA']:.4f}  "
            f"IDF1 {m['IDF1']:.4f}  IDSW {m['IDSW']:>5}  pipeline {np.median(tm):.1f}/{np.percentile(tm, 95):.1f} ms"
            + (f"  causal {causal}" if causal else "") + "  | "
            + "  ".join(f"{k} HOTA {v['HOTA']:.3f}" for k, v in res["per_class"].items()))
    print(line)
    (out / "summary.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
