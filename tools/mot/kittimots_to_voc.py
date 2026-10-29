#!/usr/bin/env python3
"""
MOT stage (2026-10-22): convert KITTI-MOTS to the VOC layout the 40 detectors already read, and count everything.

    python tools/mot/kittimots_to_voc.py                 # ~/Physical_AI/datasets/KITTI_MOTS -> KITTI_MOTS_VOC
    python tools/mot/kittimots_to_voc.py --check-only    # recount an existing conversion, write nothing

Input  (tools/mot/get_kitti_mots.sh):
    KITTI_MOTS/training/image_02/<seq>/<frame>.png
    KITTI_MOTS/instances/<seq>/<frame>.png     uint16, value = class * 1000 + instance, 0 = background,
                                               10000 = ignore region; class 1 = car, 2 = pedestrian
Output (KITTI_MOTS_VOC):
    JPEGImages/<seq>_<frame>.jpg              JPEG q95 (the VOC loader reads .jpg)
    Annotations/<seq>_<frame>.xml             one <object> per mask: tight box of the mask (1-based VOC px),
                                               <track_id> = MOTS object id, <area> = mask pixels.
                                               [Unified] each 8-connected ignore component is written twice
                                               (car and pedestrian) with <difficult>1 -> training skips it,
                                               evaluation neither rewards nor penalises a detection on it.
    ImageSets/Main/trainval.txt                MOTS train, 12 sequences. Order = round-robin over sequences
                                               (the first 16 ids of overfit_check.py then come from 12 scenes)
    ImageSets/Main/test.txt                    MOTS val, 9 sequences (2 6 7 8 10 13 14 16 18), same order rule
    ImageSets/MOT/{train,val}_seqmap.txt        sequence order for trackers (frames are read in id order)
    mots_index.json, count_report.txt          counts per split, compared with the MOTS paper (Table 1)

Expected counts (Voigtlaender et al., CVPR 2019, Table 1):
    frames 5,027 / 2,981, car tracks 431 / 151, car masks 18,831 / 8,068,
    pedestrian tracks 99 / 68, pedestrian masks 8,073 / 3,347   (train / val)
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("~/Physical_AI/datasets").expanduser()
VAL_SEQS = {2, 6, 7, 8, 10, 13, 14, 16, 18}
CLASSES = {1: "car", 2: "pedestrian"}
IGNORE_ID = 10000
EXPECTED = {
    "train": {"sequences": 12, "frames": 5027, "tracks_car": 431, "masks_car": 18831,
              "tracks_pedestrian": 99, "masks_pedestrian": 8073},
    "val": {"sequences": 9, "frames": 2981, "tracks_car": 151, "masks_car": 8068,
            "tracks_pedestrian": 68, "masks_pedestrian": 3347},
}
SCALES = {"608 letterbox (x0.490)": 608 / 1242, "1216 wide (x0.979)": 1216 / 1242}


def read_frame(args):
    """-> (frame record, objects). Writes the JPEG and XML unless check_only."""
    seq, frame, src, dst, check_only = args
    fid = f"{seq:04d}_{frame:06d}"
    img_p = src / "training" / "image_02" / f"{seq:04d}" / f"{frame:06d}.png"
    ann_p = src / "instances" / f"{seq:04d}" / f"{frame:06d}.png"
    if check_only:
        h, w = cv2.imread(str(dst / "JPEGImages" / f"{fid}.jpg"), cv2.IMREAD_GRAYSCALE).shape
    else:
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        cv2.imwrite(str(dst / "JPEGImages" / f"{fid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    objs, ignores, missing_ann = [], [], not ann_p.is_file()
    if not missing_ann:
        ann = cv2.imread(str(ann_p), cv2.IMREAD_UNCHANGED)
        assert ann.shape == (h, w), f"{fid}: annotation {ann.shape} vs image {(h, w)}"
        for v in np.unique(ann):
            v = int(v)
            if v == 0:
                continue
            m = ann == v
            if v == IGNORE_ID:
                n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                for k in range(1, n):
                    x, y, bw, bh, area = st[k]
                    ignores.append((int(x), int(y), int(x + bw - 1), int(y + bh - 1), int(area)))
                continue
            cls = v // 1000
            if cls not in CLASSES:
                raise ValueError(f"{fid}: unexpected object id {v}")
            ys, xs = np.nonzero(m)
            objs.append({"cls": CLASSES[cls], "track": v, "area": int(m.sum()),
                         "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    if not check_only:
        write_xml(dst / "Annotations" / f"{fid}.xml", fid, w, h, seq, frame, objs, ignores)
    return {"id": fid, "seq": seq, "frame": frame, "w": w, "h": h, "missing_ann": missing_ann,
            "objs": objs, "ignores": ignores}


def write_xml(path, fid, w, h, seq, frame, objs, ignores):
    r = ET.Element("annotation")
    ET.SubElement(r, "filename").text = f"{fid}.jpg"
    ET.SubElement(r, "source").text = f"KITTI-MOTS seq {seq:04d} frame {frame:06d}"
    ET.SubElement(r, "seq").text = str(seq)
    ET.SubElement(r, "frame").text = str(frame)
    sz = ET.SubElement(r, "size")
    for k, v in (("width", w), ("height", h), ("depth", 3)):
        ET.SubElement(sz, k).text = str(v)

    def add(name, box, difficult, track, area):
        o = ET.SubElement(r, "object")
        ET.SubElement(o, "name").text = name
        ET.SubElement(o, "difficult").text = str(difficult)
        ET.SubElement(o, "track_id").text = str(track)
        ET.SubElement(o, "area").text = str(area)
        bb = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bb, k).text = str(v + 1)          # VOC is 1-based; voc.py subtracts 1

    for o in objs:
        add(o["cls"], o["box"], 0, o["track"], o["area"])
    for (x0, y0, x1, y1, area) in ignores:
        for name in CLASSES.values():
            add(name, (x0, y0, x1, y1), 1, IGNORE_ID, area)
    ET.ElementTree(r).write(path)


def round_robin(frames_by_seq):
    out, lists = [], [list(v) for _, v in sorted(frames_by_seq.items())]
    while any(lists):
        for lst in lists:
            if lst:
                out.append(lst.pop(0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "KITTI_MOTS"))
    ap.add_argument("--dst", default=str(ROOT / "KITTI_MOTS_VOC"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    for d in ("JPEGImages", "Annotations", "ImageSets/Main", "ImageSets/MOT"):
        (dst / d).mkdir(parents=True, exist_ok=True)

    seqs = sorted(int(p.name) for p in (src / "training" / "image_02").iterdir() if p.is_dir())
    jobs = []
    for s in seqs:
        frames = sorted(int(p.stem) for p in (src / "training" / "image_02" / f"{s:04d}").glob("*.png"))
        jobs += [(s, f, src, dst, a.check_only) for f in frames]
    print(f"{len(seqs)} sequences, {len(jobs)} frames -> {dst}")
    with Pool(a.workers) as pool:
        recs = []
        for i, r in enumerate(pool.imap(read_frame, jobs, chunksize=16), 1):
            recs.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    split_of = lambda s: "val" if s in VAL_SEQS else "train"
    stats = {sp: Counter() for sp in ("train", "val")}
    tracks = {sp: defaultdict(set) for sp in ("train", "val")}
    heights = {sp: defaultdict(list) for sp in ("train", "val")}
    by_seq = defaultdict(list)
    seq_info = {}
    for r in recs:
        sp = split_of(r["seq"])
        by_seq[(sp, r["seq"])].append(r["id"])
        info = seq_info.setdefault(r["seq"], {"split": sp, "frames": 0, "width": r["w"], "height": r["h"]})
        info["frames"] += 1
        st = stats[sp]
        st["frames"] += 1
        st["frames_missing_annotation"] += r["missing_ann"]
        st["frames_without_objects"] += len(r["objs"]) == 0
        st["ignore_regions"] += len(r["ignores"])
        for o in r["objs"]:
            st[f"masks_{o['cls']}"] += 1
            tracks[sp][o["cls"]].add((r["seq"], o["track"]))
            heights[sp][o["cls"]].append(o["box"][3] - o["box"][1] + 1)
    for sp in stats:
        stats[sp]["sequences"] = len({s for (p, s) in by_seq if p == sp})
        for c in CLASSES.values():
            stats[sp][f"tracks_{c}"] = len(tracks[sp][c])

    if not a.check_only:
        for sp, name in (("train", "trainval"), ("val", "test")):
            per_seq = {s: ids for (p, s), ids in by_seq.items() if p == sp}
            (dst / "ImageSets" / "Main" / f"{name}.txt").write_text("\n".join(round_robin(per_seq)) + "\n")
            (dst / "ImageSets" / "MOT" / f"{sp}_seqmap.txt").write_text(
                "\n".join(f"{s:04d} {len(per_seq[s])}" for s in sorted(per_seq)) + "\n")

    lines = ["KITTI-MOTS -> VOC count report", "",
             f"{'item':<28}{'train':>9}{'exp':>9}{'':>6}{'val':>9}{'exp':>9}{'':>6}"]
    ok_all = True
    keys = ["sequences", "frames", "tracks_car", "masks_car", "tracks_pedestrian", "masks_pedestrian",
            "ignore_regions", "frames_without_objects", "frames_missing_annotation"]
    for k in keys:
        row = f"{k:<28}"
        for sp in ("train", "val"):
            got, exp = stats[sp][k], EXPECTED[sp].get(k)
            flag = "" if exp is None else ("OK" if got == exp else "DIFF")
            ok_all &= flag != "DIFF"
            row += f"{got:>9,}{'' if exp is None else f'{exp:,}':>9}{flag:>6}"
        lines.append(row)
    lines += ["", "box height (px) after scaling - share of objects below 16 / 32 px:"]
    for sp in ("train", "val"):
        for c in CLASSES.values():
            hgt = np.array(heights[sp][c], dtype=np.float32)
            if not len(hgt):
                continue
            parts = [f"{nm}: <16 {np.mean(hgt * s < 16):.1%}, <32 {np.mean(hgt * s < 32):.1%}"
                     for nm, s in SCALES.items()]
            lines.append(f"  {sp:<5} {c:<10} n={len(hgt):>6,}  median {np.median(hgt):5.0f} px  |  " + "  |  ".join(parts))
    sizes = Counter((v["width"], v["height"]) for v in seq_info.values())
    lines += ["", f"image sizes: {dict(sizes)}", "", "RESULT " + ("OK" if ok_all else "DIFF (see rows marked DIFF)")]
    report = "\n".join(lines)
    print("\n" + report)
    (dst / "count_report.txt").write_text(report + "\n")
    (dst / "mots_index.json").write_text(json.dumps(
        {"source": str(src), "val_sequences": sorted(VAL_SEQS), "sequences": seq_info,
         "counts": {sp: dict(st) for sp, st in stats.items()}, "expected": EXPECTED}, indent=1))


if __name__ == "__main__":
    main()
