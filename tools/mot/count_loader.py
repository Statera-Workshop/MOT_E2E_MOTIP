#!/usr/bin/env python3
"""
MOT stage (2026-10-22): count what the detector loader reads, and compare with the converter's own count.

    UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py

Lesson of 10-16 (FL36 letters silently dropped by a case mismatch): the generator's count is not enough; the
numbers must come from common/datasets/voc.py itself. Checks
    1. images per split = frames in mots_index.json
    2. objects per class read by the loader (non-difficult) = masks per class
    3. difficult objects per class = ignore regions (each region is written once per class)
    4. every class index of config.VOC_CLASSES is used in both splits
    5. one training sample per split decodes to targets with class ids inside [0, C)
Prints RESULT OK / RESULT FAIL and exits 1 on failure (the GPU runner stops before overfit checks).
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.datasets.voc import VOCDetection  # noqa: E402


def main():
    if C.IMAGE_DATASET != "kittimots":
        sys.exit("set UNIFIED_IMAGE_DATASET=kittimots")
    index = json.loads((C.VOC_ROOT / "mots_index.json").read_text())
    names = list(C.VOC_CLASSES)
    ok = True
    print(f"root {C.VOC_ROOT}  classes {names}")
    for split, sp in (("trainval", "train"), ("test", "val")):
        ds = VOCDetection(C.VOC_ROOT, split, 608, names, train=False)
        exp = index["counts"][sp]
        objs, ign = Counter(), Counter()
        for i in range(len(ds)):
            a = ds.annotation(i)
            for lab, dif in zip(a["labels"], a["ignore"]):
                (ign if dif else objs)[names[int(lab)]] += 1
        rows = [("images", len(ds), exp["frames"])]
        for c in names:
            rows.append((f"objects {c}", objs[c], exp.get(f"masks_{c}", 0)))
            rows.append((f"ignore  {c}", ign[c], exp.get("ignore_regions", 0)))
        print(f"\n[{split} = MOTS {sp}]")
        for k, got, want in rows:
            flag = "OK" if got == want else "FAIL"
            ok &= flag == "OK"
            print(f"  {k:<20}{got:>9,}  converter {want:>9,}  {flag}")
        unused = [c for c in names if objs[c] == 0]
        if unused:
            ok = False
            print(f"  FAIL: classes never read: {unused}")
        _, _, t = ds.__class__(C.VOC_ROOT, split, 608, names, train=True, max_samples=1)[0]
        cls = t[:, 1].long().tolist() if len(t) else []
        bad = [c for c in cls if not 0 <= c < len(names)]
        print(f"  sample 0 targets {tuple(t.shape)} classes {sorted(set(cls))}" + ("  FAIL" if bad else ""))
        ok &= not bad
    print("\nRESULT " + ("OK" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
