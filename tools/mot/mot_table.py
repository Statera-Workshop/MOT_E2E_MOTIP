#!/usr/bin/env python3
"""
MOT stage (2026-10-23): collect ~/mot_logs/track/kittimots_*/<tracker>.json into markdown tables.

    python tools/mot/mot_table.py                 # main table (official defaults) + threshold sensitivity
    python tools/mot/mot_table.py --metric IDF1

Main table: one row per detector, one column per tracker, metric with official default parameters; a detector with
both <v>.npz and <v>_reid.npz results uses the _reid run (same detections, plus the ReID trackers).
Latency column: detector + (ReID, for appearance trackers) + tracker median ms per frame.
Sensitivity table: the same metric with one common score threshold for every tracker (analysis only, not ranking:
the threshold is chosen on val).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ORDER = ["iou", "sort", "bytetrack", "ocsort", "hybridsort", "botsort", "deepsort", "botsort_reid", "tracktrack",
         "fairmot", "joint"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/mot_logs/track")
    ap.add_argument("--metric", default="HOTA")
    a = ap.parse_args()
    main_t, sweep = defaultdict(dict), defaultdict(lambda: defaultdict(dict))
    lat = defaultdict(dict)
    for js in sorted(Path(a.root).expanduser().glob("kittimots_*/*.json")):
        r = json.loads(js.read_text())
        run = js.parent.name[len("kittimots_"):]
        if run.startswith("oracle"):
            continue
        thr = r.get("score_thr")
        det = run.split("_thr")[0]
        reid = det.endswith("_reid")
        det = det[:-5] if reid else det
        val = r["mean"][a.metric]
        if thr is None:
            if reid or r["tracker"] not in main_t[det]:
                main_t[det][r["tracker"]] = (val, r.get("causal_check"))
                ms = (r["detector_ms"] or {}).get("median", 0) + (r.get("reid_ms") or {}).get("median", 0) \
                    + r["tracker_ms"]["median"]
                lat[det][r["tracker"]] = ms
        else:
            sweep[r["tracker"]][det][thr] = val
    trk = [t for t in ORDER if any(t in v for v in main_t.values())]
    print(f"### {a.metric}, official default parameters (KITTI-MOTS val, car / pedestrian mean)\n")
    print("| detector | " + " | ".join(trk) + " | best |")
    print("|---" * (len(trk) + 2) + "|")
    for det in sorted(main_t, key=lambda d: -max(v[0] for v in main_t[d].values())):
        row = main_t[det]
        best = max(row, key=lambda t: row[t][0])
        cells = []
        for t in trk:
            if t not in row:
                cells.append("-")
                continue
            v, c = row[t]
            s = f"{v:.3f}" + ("" if c in (None, "OK") else " (causal FAIL)")
            cells.append(f"**{s}**" if t == best else s)
        print(f"| {det} | " + " | ".join(cells) + f" | {best} |")
    print("\n### pipeline latency per frame, ms (detector + ReID + tracker, medians)\n")
    print("| detector | " + " | ".join(trk) + " |")
    print("|---" * (len(trk) + 1) + "|")
    for det in sorted(lat):
        print(f"| {det} | " + " | ".join(f"{lat[det][t]:.1f}" if t in lat[det] else "-" for t in trk) + " |")
    if sweep:
        print(f"\n### {a.metric} vs common score threshold (sensitivity; threshold seen on val -> not for ranking)\n")
        for t in [t for t in ORDER if t in sweep]:
            thrs = sorted({th for d in sweep[t].values() for th in d})
            print(f"\n{t}\n\n| detector | " + " | ".join(f"{th:.1f}" for th in thrs) + " |")
            print("|---" * (len(thrs) + 1) + "|")
            for det in sorted(sweep[t]):
                print(f"| {det} | " + " | ".join(f"{sweep[t][det].get(th, float('nan')):.3f}" for th in thrs) + " |")


if __name__ == "__main__":
    main()
