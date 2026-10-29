#!/usr/bin/env python3
"""
MOT stage (2026-10-23): run online trackers on saved per-frame detections and evaluate them.

    # oracle boxes (GT + noise, drop-outs, false positives) - checks trackers and evaluation without a detector
    python tools/mot/run_tracker.py --source fl36mot --oracle --trackers iou sort bytetrack ocsort --causal
    python tools/mot/run_tracker.py --source fl36mot --gt-as-pred        # evaluation self-test, must print OK
    python tools/mot/run_tracker.py --source fl36mot --oracle-exact      # exact boxes: upper bound per tracker
    # detections saved by tools/mot/dump_dets.py
    python tools/mot/run_tracker.py --source kittimots --dets ~/mot_logs/dets/kittimots/x.npz --trackers bytetrack

Detections npz: one array per sequence, key = sequence name, rows [frame, x1, y1, x2, y2, score, class]
(class index into the dataset classes), plus "<seq>__ms" (detector ms per frame) and "meta" (json string).
Output: --out/<run>/<tracker>.json (metrics per class + mean, tracker ms per frame median / p95, settings) and
        --out/<run>/summary.txt (one line per tracker). Tracker time = wall time of tracker.update() on the CPU.
--causal: for every sequence, the tracker is also run on 3 random prefixes; the last frame of each prefix must give
exactly the output of the full run at that frame (online methods only).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.core import GMCSparseOptFlow  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, evaluate, load_fl36mot, load_kittimots  # noqa: E402
from common.mot.trackers import TRACKERS, make_tracker  # noqa: E402

FL36_CLASSES = tuple(C.FL36_CLASSES)


def load_gt(source, split):
    """-> gt, class names, frame rate, frame image paths per sequence"""
    if source == "kittimots":
        root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
        gt, ids = load_kittimots(root, split)
        return gt, KITTI_CLASSES, 10.0, {s: [str(root / "JPEGImages" / f"{i}.jpg") for i in v] for s, v in ids.items()}
    gt, paths = load_fl36mot(C.DATASETS_ROOT / "FL36_MOT", split)
    return gt, FL36_CLASSES, 10.0, paths


def oracle_dets(gt, rng, exact=False, sigma=0.04, drop=0.10, fp_rate=0.3, feat_dim=64, feat_noise=0.6):
    """GT boxes as detections. Not exact: jitter (sigma x box size), random misses, false positives, scores.
    Also appearance features for the ReID trackers: a fixed random unit vector per GT id plus Gaussian noise
    (feat_noise per dimension x 1/sqrt(dim); exact: no noise); false positives get random vectors."""
    out, feats_out, proto = {}, {}, {}
    for seq, frames in gt.items():
        per, pf = [], []
        for f, g in enumerate(frames):
            b, c, ids = g["boxes"].copy(), g["cls"], g["ids"]
            for i in ids:
                if (seq, i) not in proto:
                    v = rng.normal(size=feat_dim)
                    proto[(seq, i)] = v / np.linalg.norm(v)
            fe = np.array([proto[(seq, i)] for i in ids]).reshape(-1, feat_dim)
            if exact:
                per.append(np.c_[np.full(len(b), f), b, np.ones(len(b)), c])
                pf.append(fe)
                continue
            keep = rng.random(len(b)) > drop
            b, c, fe = b[keep], c[keep], fe[keep]
            pf.append(fe + rng.normal(0, feat_noise / np.sqrt(feat_dim), fe.shape))
            wh = np.c_[b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
            b = b + rng.normal(0, sigma, b.shape) * wh
            s = np.clip(rng.normal(0.75, 0.15, len(b)), 0.05, 1.0)
            rows = [np.c_[np.full(len(b), f), b, s, c]]
            nfp = rng.poisson(fp_rate)
            if nfp:
                xy = rng.uniform(0, 500, (nfp, 2))
                wh2 = rng.uniform(20, 100, (nfp, 2))
                rows.append(np.c_[np.full(nfp, f), xy, xy + wh2, rng.uniform(0.1, 0.7, nfp),
                                  rng.integers(0, max(int(c.max()) + 1, 1) if len(c) else 1, nfp)])
                pf.append(rng.normal(size=(nfp, feat_dim)) / np.sqrt(feat_dim))
            per.append(np.concatenate(rows))
        out[seq] = np.concatenate(per) if per else np.zeros((0, 7))
        feats_out[seq] = np.concatenate(pf) if pf else np.zeros((0, feat_dim))
    return out, feats_out


def run_sequence(name, rows, n_frames, frame_rate, upto=None, feats=None, paths=None, score_thr=None):
    """Tracker time = tracker.update() + (BoT-SORT) camera-motion estimation; image reading is not timed."""
    import cv2
    trk = make_tracker(name, frame_rate=frame_rate, score_thr=score_thr)
    gmc = GMCSparseOptFlow() if trk.needs_gmc else None
    preds, times = [], []
    for f in range(n_frames if upto is None else upto):
        sel = rows[:, 0] == f
        d = rows[sel][:, 1:7]
        fe = feats[sel] if feats is not None else None
        img = cv2.imread(paths[f]) if gmc is not None else None
        t0 = time.perf_counter()
        warp = gmc.apply(img) if gmc is not None else None
        o = trk.update(d, feats=fe, warp=warp)
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(o)
    return preds, times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fl36mot", "kittimots"], required=True)
    ap.add_argument("--split", default="val")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dets", help="npz from dump_dets.py")
    g.add_argument("--oracle", action="store_true")
    g.add_argument("--oracle-exact", action="store_true")
    g.add_argument("--gt-as-pred", action="store_true", help="evaluation self-test: GT ids as predictions -> 1.0")
    ap.add_argument("--trackers", nargs="+", default=list(TRACKERS))
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--score-thr", type=float, default=None,
                    help="sensitivity analysis: same main score threshold for every tracker (not for ranking)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=str(Path("~/mot_logs/track").expanduser()))
    ap.add_argument("--run", default=None, help="run name (default: from --dets or oracle)")
    a = ap.parse_args()

    gt, classes, fps, paths = load_gt(a.source, a.split)
    rng = np.random.default_rng(a.seed)
    meta = {}
    if a.gt_as_pred:
        pred = {seq: [np.c_[g["boxes"], g["ids"], g["cls"], np.ones(len(g["ids"]))] for g in frames]
                for seq, frames in gt.items()}
        m = evaluate(gt, pred, classes)["mean"]
        ok = all(abs(m[k] - 1.0) < 1e-9 for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")) and m["IDSW"] == 0
        print(f"gt-as-pred: HOTA {m['HOTA']:.6f} DetA {m['DetA']:.6f} AssA {m['AssA']:.6f} MOTA {m['MOTA']:.6f} "
              f"IDF1 {m['IDF1']:.6f} IDSW {m['IDSW']}  -> {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    if a.dets:
        z = np.load(Path(a.dets).expanduser(), allow_pickle=False)
        dets = {seq: z[seq] for seq in gt}
        feats = {seq: z[f"{seq}__feat"].astype(np.float32) for seq in gt} if all(f"{seq}__feat" in z for seq in gt) \
            else None
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        det_ms = np.concatenate([z[f"{seq}__ms"] for seq in gt if f"{seq}__ms" in z]) if meta else np.zeros(0)
        run = a.run or f"{a.source}_{Path(a.dets).stem}"
    else:
        dets, feats = oracle_dets(gt, rng, exact=a.oracle_exact)
        det_ms = np.zeros(0)
        run = a.run or f"{a.source}_oracle{'_exact' if a.oracle_exact else ''}"
    if a.score_thr is not None:
        run += f"_thr{a.score_thr:.2f}"
    reid_ms = np.concatenate([z[f"{seq}__reid_ms"] for seq in gt]) if a.dets and all(
        f"{seq}__reid_ms" in z for seq in gt) else np.zeros(0)
    out = Path(a.out).expanduser() / run
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run}: {len(gt)} sequences, {sum(len(v) for v in gt.values())} frames, "
          f"{sum(len(v) for v in dets.values())} detections, classes {len(classes)}")
    lines = []
    for name in a.trackers:
        if make_tracker(name).needs_feats and feats is None:
            print(f"{name:<12} skipped: no appearance features in {a.dets} (tools/mot/dump_reid.py)")
            continue
        pred, times = {}, []
        for seq, frames in gt.items():
            pred[seq], t = run_sequence(name, dets[seq], len(frames), fps, feats=feats[seq] if feats else None,
                                        paths=paths[seq], score_thr=a.score_thr)
            times += t
        causal = None
        if a.causal:
            causal = "OK"
            for seq, frames in gt.items():
                for upto in sorted(rng.integers(1, len(frames) + 1, 3)):
                    p, _ = run_sequence(name, dets[seq], len(frames), fps, upto=upto,
                                        feats=feats[seq] if feats else None, paths=paths[seq],
                                        score_thr=a.score_thr)
                    if not np.array_equal(p[-1], pred[seq][upto - 1]):
                        causal = f"FAIL ({seq} frame {upto - 1})"
                        break
                if causal != "OK":
                    break
        res = evaluate(gt, pred, classes)
        tm = np.array(times)
        rec = {"tracker": name, "run": run, "source": a.source, "split": a.split, "detections": meta,
               "tracker_ms": {"median": float(np.median(tm)), "p95": float(np.percentile(tm, 95)),
                              "frames": len(tm)},
               "detector_ms": {"median": float(np.median(det_ms)), "p95": float(np.percentile(det_ms, 95))}
               if len(det_ms) else None,
               "reid_ms": {"median": float(np.median(reid_ms)), "p95": float(np.percentile(reid_ms, 95))}
               if len(reid_ms) and make_tracker(name).needs_feats else None,
               "score_thr": a.score_thr, "causal_check": causal, **res}
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        m = res["mean"]
        line = (f"{name:<12} HOTA {m['HOTA']:.4f}  DetA {m['DetA']:.4f}  AssA {m['AssA']:.4f}  "
                f"MOTA {m['MOTA']:.4f}  IDF1 {m['IDF1']:.4f}  IDSW {m['IDSW']:>5}  "
                f"trk {rec['tracker_ms']['median']:.2f}/{rec['tracker_ms']['p95']:.2f} ms"
                + (f"  det {rec['detector_ms']['median']:.1f} ms" if rec["detector_ms"] else "")
                + (f"  reid {rec['reid_ms']['median']:.1f} ms" if rec["reid_ms"] else "")
                + (f"  causal {causal}" if causal else ""))
        if len(classes) <= 4:
            line += "  | " + "  ".join(f"{k} HOTA {v['HOTA']:.3f}" for k, v in res["per_class"].items())
        print(line, flush=True)
        lines.append(line)
    with open(out / "summary.txt", "a") as fh:                       # appended: later runs add trackers
        fh.write(f"{run}  ({time.strftime('%F %T')})\n" + "\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
