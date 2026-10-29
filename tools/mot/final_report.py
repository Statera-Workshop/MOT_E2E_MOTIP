#!/usr/bin/env python3
"""
MOT stage (2026-10-29): final tables of the KITTI-MOTS tracking comparison, from the evaluation JSON files.

    python tools/mot/final_report.py            # -> ~/mot_logs/final_report.md (+ final_report.json)

Sections: 1 main table (HOTA, official default parameters, every detector x tracker, JDT / E2E rows);
2 tracker summary over the detectors (mean HOTA, mean rank, wins); 3 speed-accuracy Pareto front over every
combination (pipeline ms = detector + ReID + tracker medians); 4 JDT / E2E against tracking-by-detection on the same
backbone / detector; 5 MOTIP training curve; 6 score-threshold sensitivity (best common threshold per detector,
analysis only); 7 model size and FLOPs (tools/mot/model_stats.py). Only protocol values enter the main table:
MOTIP = last epoch (61); intermediate epochs only in section 5.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("~/mot_logs").expanduser()
TRK = ["iou", "sort", "bytetrack", "ocsort", "hybridsort", "botsort", "deepsort", "botsort_reid", "tracktrack", "fairmot"]
JOINT = {"centertrack": ("JDT", "centernet"), "fairmot": ("JDT", "centernet"), "trades": ("JDT", "centernet"),
         "motip_e61": ("E2E", "deformable_detr")}
MOTIP_CURVE = [("motip", 5), ("motip_e13", 13), ("motip_e25", 25), ("motip_e37", 37), ("motip_e49", 49), ("motip_e61", 61)]


def load():
    main, sweep, joint = defaultdict(dict), defaultdict(lambda: defaultdict(dict)), {}
    for js in sorted((ROOT / "track").glob("kittimots_*/*.json")):
        r = json.loads(js.read_text())
        run = js.parent.name[len("kittimots_"):]
        if run.startswith("oracle"):
            continue
        m = r["mean"]
        ms = (r.get("detector_ms") or {}).get("median", 0) + (r.get("reid_ms") or {}).get("median", 0) \
            + r["tracker_ms"]["median"]
        rec = {"HOTA": m["HOTA"], "DetA": m["DetA"], "AssA": m["AssA"], "MOTA": m["MOTA"], "IDF1": m["IDF1"],
               "IDSW": m["IDSW"], "ms": ms, "causal": r.get("causal_check"),
               "car": r["per_class"].get("car", {}).get("HOTA"), "ped": r["per_class"].get("pedestrian", {}).get("HOTA")}
        if r["tracker"] == "joint":
            joint[run] = rec
            continue
        det = run.split("_thr")[0]
        reid = det.endswith("_reid")
        det = det[:-5] if reid else det
        if r.get("score_thr") is None:
            if reid or r["tracker"] not in main[det]:
                main[det][r["tracker"]] = rec
        else:
            sweep[det][r["tracker"]][r["score_thr"]] = m["HOTA"]
    return main, sweep, joint


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    main_t, sweep, joint = load()
    if "fairmot" in main_t and "fairmot" in main_t["fairmot"]:                # FairMOT = its network + its association
        joint["fairmot"] = main_t["fairmot"]["fairmot"]
    dets = sorted(main_t, key=lambda d: -max(v["HOTA"] for v in main_t[d].values()))
    L = ["# KITTI-MOTS val tracking comparison (box level, online)", ""]
    # 1
    L += ["## 1. HOTA, official default parameters", "",
          "| detector | " + " | ".join(TRK) + " | best |", "|---" * (len(TRK) + 2) + "|"]
    for d in dets:
        row = main_t[d]
        best = max(row, key=lambda t: row[t]["HOTA"])
        cells = [("**" + fmt(row[t]["HOTA"]) + "**" if t == best else fmt(row[t]["HOTA"])) if t in row else "-"
                 for t in TRK]
        L.append(f"| {d} | " + " | ".join(cells) + f" | {best} |")
    for name, (fam, _) in JOINT.items():
        if name in joint and name != "fairmot":                                # FairMOT already has its own row
            L.append(f"| {name} ({fam}) | " + " | ".join("-" for _ in TRK) + f" | {fmt(joint[name]['HOTA'])} |")
    bad = [(d, t) for d in main_t for t, v in main_t[d].items() if v["causal"] not in (None, "OK")]
    L += ["", f"Causality check: {'all OK' if not bad else 'FAILED for ' + str(bad)}.", ""]
    # 2
    L += ["## 2. Trackers over all detectors", "", "| tracker | detectors | mean HOTA | mean rank | wins |",
          "|---|---|---|---|---|"]
    ranks, wins = defaultdict(list), defaultdict(int)
    for d in dets:
        present = [t for t in TRK if t in main_t[d]]
        order = sorted(present, key=lambda t: -main_t[d][t]["HOTA"])
        for i, t in enumerate(order):
            ranks[t].append(i + 1)
        wins[order[0]] += 1
    summ = []
    for t in TRK:
        vals = [main_t[d][t]["HOTA"] for d in dets if t in main_t[d]]
        if vals:
            summ.append((t, len(vals), float(np.mean(vals)), float(np.mean(ranks[t])), wins[t]))
    for t, n, mh, mr, w in sorted(summ, key=lambda s: s[3]):
        L.append(f"| {t} | {n} | {mh:.3f} | {mr:.2f} | {w} |")
    L.append("")
    # 3
    combos = [(f"{d} + {t}", v["ms"], v["HOTA"]) for d in main_t for t, v in main_t[d].items()
              if not (d == "fairmot" and t == "fairmot")]
    combos += [(f"{n} ({JOINT[n][0]})", joint[n]["ms"], joint[n]["HOTA"]) for n in JOINT if n in joint]
    front, best = [], -1
    for name, ms, h in sorted(combos, key=lambda c: (c[1], -c[2])):
        if h > best:
            front.append((name, ms, h))
            best = h
    L += ["## 3. Speed-accuracy Pareto front (RTX 4070 Laptop, batch 1, fp32, 1216 x 384)", "",
          "| combination | ms / frame | HOTA |", "|---|---|---|"] + [f"| {n} | {ms:.1f} | {h:.3f} |" for n, ms, h in front] + [""]
    # 4
    L += ["## 4. Joint / end-to-end against tracking-by-detection on the same network", "",
          "| method | family | HOTA | DetA | AssA | ms | same-network TbD best | HOTA |", "|---|---|---|---|---|---|---|---|"]
    for n, (fam, base) in JOINT.items():
        if n in joint and base in main_t:
            bt = max(main_t[base], key=lambda t: main_t[base][t]["HOTA"])
            j = joint[n]
            L.append(f"| {n} | {fam} | {fmt(j['HOTA'])} | {fmt(j['DetA'])} | {fmt(j['AssA'])} | {j['ms']:.1f} | "
                     f"{base} + {bt} | {fmt(main_t[base][bt]['HOTA'])} |")
    L.append("")
    # 5
    L += ["## 5. MOTIP training curve (val HOTA; the table uses the last epoch)", "",
          "| epochs | " + " | ".join(str(e) for _, e in MOTIP_CURVE) + " |", "|---" * (len(MOTIP_CURVE) + 1) + "|",
          "| HOTA | " + " | ".join(fmt(joint.get(n, {}).get("HOTA")) for n, _ in MOTIP_CURVE) + " |", ""]
    # 6
    if sweep:
        L += ["## 6. Best common score threshold per detector (sensitivity; chosen on val -> not used for ranking)", "",
              "| detector | " + " | ".join(t for t in TRK if any(t in sweep[d] for d in sweep)) + " |"]
        ts = [t for t in TRK if any(t in sweep[d] for d in sweep)]
        L.append("|---" * (len(ts) + 1) + "|")
        for d in sorted(sweep):
            cells = []
            for t in ts:
                s = sweep[d].get(t)
                cells.append(f"{max(s, key=s.get):.1f} ({max(s.values()):.3f})" if s else "-")
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        L.append("")
    # 7
    ms_path = ROOT / "model_stats.md"
    if ms_path.exists():
        L += ["## 7. Model size and computation", "", ms_path.read_text()]
    (ROOT / "final_report.md").write_text("\n".join(L) + "\n")
    (ROOT / "final_report.json").write_text(json.dumps({"main": main_t, "joint": joint, "pareto": front,
                                                        "tracker_summary": summ}, indent=1, default=str))
    print("\n".join(L[:60]))
    print(f"... -> {ROOT / 'final_report.md'}")


if __name__ == "__main__":
    main()
