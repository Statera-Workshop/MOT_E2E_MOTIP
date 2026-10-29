#!/usr/bin/env python3
"""
MOT stage (2026-10-22): build KITTI-MOTS style instance PNGs from the KITTI-STEP panoptic maps.

Why: the KITTI-MOTS site (vision.rwth-aachen.de) refuses connections from this network. KITTI-STEP has the same
21 training sequences and the same 12 / 9 split, and its car / person instances were merged from the KITTI-MOTS
tracking ground truth (deeplab2 g3doc/setup/kitti_step.md). [Unified] data source = KITTI-STEP, recorded in the
progress tab; the count report still compares with the MOTS paper and shows any difference as DIFF.

    wget -c http://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz   (in datasets/downloads)
    tar -xzf kitti-step.tar.gz -C ~/Physical_AI/datasets/                               -> datasets/kitti-step/
    python tools/mot/step_to_mots_png.py [--step ~/Physical_AI/datasets/kitti-step]

STEP panoptic PNG: R = semantic id (Cityscapes 19 classes, person 11, car 13, void 255),
                   G * 256 + B = instance id (0 = no instance)
Written          : datasets/KITTI_MOTS/instances/<seq>/<frame>.png, uint16, the KITTI-MOTS encoding
                   car 1 * 1000 + id, pedestrian 2 * 1000 + id, 10000 = ignore, 0 = background
                   [Unified] car / person pixels without an instance id become ignore (10000).
Instance ids >= 1000 cannot be written in this encoding -> the script stops instead of wrapping them.
"""

import argparse
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("~/Physical_AI/datasets").expanduser()
SEM_TO_MOTS = {13: 1, 11: 2}          # STEP car -> MOTS car, STEP person -> MOTS pedestrian


def convert(args):
    src, dst = args
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)                  # BGR: [..., 2] = R
    sem = img[..., 2].astype(np.int32)
    inst = img[..., 1].astype(np.int32) * 256 + img[..., 0].astype(np.int32)
    out = np.zeros(sem.shape, np.uint16)
    c = Counter()
    for s, cls in SEM_TO_MOTS.items():
        m = sem == s
        has = m & (inst > 0)
        if has.any():
            if inst[has].max() >= 1000:
                return src, None, f"instance id {int(inst[has].max())} >= 1000"
            out[has] = cls * 1000 + inst[has]
            c[f"objects_{cls}"] += len(np.unique(inst[has]))
        crowd = m & (inst == 0)
        if crowd.any():
            out[crowd] = 10000
            c["frames_with_ignore"] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    return src, c, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default=str(ROOT / "kitti-step"))
    ap.add_argument("--out", default=str(ROOT / "KITTI_MOTS" / "instances"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    step, out = Path(a.step).expanduser(), Path(a.out).expanduser()
    # panoptic_maps/{train,val}/<seq>/<frame>.png ; accept any depth, sequence = parent folder name
    pngs = sorted(p for p in step.rglob("*.png") if p.parent.name.isdigit() and "test" not in p.parts)
    seqs = sorted({p.parent.name for p in pngs})
    print(f"STEP maps: {len(pngs)} frames in {len(seqs)} sequences under {step}")
    if len(seqs) != 21:
        raise SystemExit(f"expected 21 training sequences (train 12 + val 9), found {len(seqs)}: {seqs}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty - remove it first if you want to rebuild it")
    jobs = [(p, out / f"{int(p.parent.name):04d}" / p.name) for p in pngs]
    total, errors = Counter(), []
    with Pool(a.workers) as pool:
        for i, (src, c, err) in enumerate(pool.imap(convert, jobs, chunksize=16), 1):
            if err:
                errors.append(f"{src}: {err}")
            else:
                total.update(c)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    if errors:
        print("\n".join(errors[:10]))
        raise SystemExit(f"FAILED: {len(errors)} frames could not be encoded")
    print(f"written {len(jobs)} PNGs to {out}")
    print(f"per-frame object sightings: car {total['objects_1']:,}  pedestrian {total['objects_2']:,}"
          f"  frames with ignore pixels {total['frames_with_ignore']:,}")
    (out.parent / "INSTANCES_FROM_KITTI_STEP").write_text(f"source {step}\n")


if __name__ == "__main__":
    main()
