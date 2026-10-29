#!/usr/bin/env python3
"""
MOT stage (2026-10-24): train MOTIP (common/mot/motip.py) on KITTI-MOTS train, starting from our KITTI-MOTS
Deformable DETR. Recipe of the official train.py with the reductions recorded in common/mot/motip.py.

    UNIFIED_IMAGE_H=384 python tools/mot/train_motip.py                 # 5 epochs x 1000 clips
    UNIFIED_IMAGE_H=384 python tools/mot/train_motip.py --check         # 3 clips, shapes, losses, memory

Per clip (30 frames): 4 random frames through DETR with gradient (one at a time, activation checkpointing), the other
26 without gradient (chunks of 4); DETR set loss on all 30 frames (official); matched query outputs -> trajectory
modeling (trajectory side) + ID decoder over 6 augmentation groups; loss = DETR loss + ID loss (weight 1).
AdamW 1e-4 (backbone x0.1, sampling_offsets / reference_points x0.05), weight decay 5e-4, grad-norm clip 0.1 for
DETR and for the rest separately, warm-up 200 iterations, x0.1 after epochs 3 and 4. Detector BN frozen.
Output: <study>/runs/motip/motip_kittimots_e005.pth (+ motip_last.pth after every epoch).
"""
import argparse
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.evaluate import load_kittimots  # noqa: E402
from common.mot.motip import CLIP_LEN, ClipDataset, MOTIPNet, id_loss, prepare_seq_info  # noqa: E402
from common.utils import load_checkpoint  # noqa: E402
from detectors.deformable_detr.build import DeformableDETR  # noqa: E402


def detr_outputs(model, imgs, n_grad=4, chunk=4):
    T = imgs.shape[0]
    perm = torch.randperm(T)
    outs = [None] * T
    with torch.no_grad():
        ng = perm[n_grad:].tolist()
        for k in range(0, len(ng), chunk):
            idx = ng[k:k + chunk]
            o = model.detr.heads(imgs[idx])
            for j, t in enumerate(idx):
                outs[t] = {"pred_logits": o["pred_logits"][j:j + 1], "pred_boxes": o["pred_boxes"][j:j + 1],
                           "hs": o["hs"][j:j + 1],
                           "aux_outputs": [{"pred_logits": a["pred_logits"][j:j + 1], "pred_boxes": a["pred_boxes"][j:j + 1]}
                                           for a in o["aux_outputs"]]}
    for t in perm[:n_grad].tolist():
        outs[t] = checkpoint(model.detr.heads, imgs[t:t + 1], use_reentrant=False)
    cat = lambda k: torch.cat([o[k] for o in outs])
    batched = {"pred_logits": cat("pred_logits"), "pred_boxes": cat("pred_boxes"),
               "aux_outputs": [{"pred_logits": torch.cat([o["aux_outputs"][l]["pred_logits"] for o in outs]),
                                "pred_boxes": torch.cat([o["aux_outputs"][l]["pred_boxes"] for o in outs])}
                               for l in range(len(outs[0]["aux_outputs"]))]}
    return batched, [o["hs"][0] for o in outs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detr", default=str(C.RUNS_ROOT / "image_deformable_detr" / "checkpoints" /
                                          "image_deformable_detr_kittimots_e030.pth"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--clips", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=1216)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--init", default=None, help="continue from a MOTIP checkpoint (model weights only)")
    ap.add_argument("--milestones", default="3,4", help="epochs after which the lr is multiplied by 0.1")
    ap.add_argument("--max-ids-check", action="store_true", help="check on the clip with the most identities")
    ap.add_argument("--out", default=str(C.RUNS_ROOT / "motip" / "motip_kittimots_e005.pth"))
    a = ap.parse_args()
    assert os.environ.get("UNIFIED_IMAGE_H") == "384", "set UNIFIED_IMAGE_H=384 (wide input)"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda")
    root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
    gt, ids = load_kittimots(root, "train")
    paths = {s: [str(root / "JPEGImages" / f"{i}.jpg") for i in v] for s, v in ids.items()}
    ds = ClipDataset(gt, paths, a.img_size, clips_per_epoch=3 if a.check else a.clips, aug=C.PROFILE["image_aug"],
                     hflip_prob=C.PROFILE["hflip_prob"], seed=a.seed)
    if a.check:                                                       # make the check see the densest clips
        dens = sorted(ds.starts, key=lambda sf: -len({int(i) for f in range(sf[1], sf[1] + CLIP_LEN)
                                                      for i in gt[sf[0]][f]["ids"]}))
        ds.starts = dens[:3]
    print(f"clips: {len(ds.starts)} possible starts, {len(ds)} per epoch, clip length {CLIP_LEN}", flush=True)
    detr = DeformableDETR(2)
    load_checkpoint(Path(a.detr), detr.to(device), device)
    model = MOTIPNet(detr).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["model_state_dict"])
        print(f"continue from {a.init}", flush=True)
    model.freeze_detr_bn()                                          # before the optimizer sees the params
    crit = detr.criterion
    groups = {"backbone": [], "proj": [], "detr": [], "other": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("detr.backbone"):
            groups["backbone"].append(p)
        elif "sampling_offsets" in n or "reference_points" in n:
            groups["proj"].append(p)
        elif n.startswith("detr."):
            groups["detr"].append(p)
        else:
            groups["other"].append(p)
    scale = {"backbone": 0.1, "proj": 0.05, "detr": 1.0, "other": 1.0}
    opt = torch.optim.AdamW([{"params": v, "lr": a.lr * scale[k], "base": a.lr * scale[k]} for k, v in groups.items()],
                            lr=a.lr, weight_decay=5e-4)
    detr_params = [p for k in ("backbone", "proj", "detr") for p in groups[k]]
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=a.workers, collate_fn=lambda b: b[0],
                                     worker_init_fn=lambda w: np.random.seed(a.seed * 100 + w + int(time.time()) % 1000))
    it, t_all = 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        model.freeze_detr_bn()
        t0, sums, n = time.time(), {"detr": 0.0, "id": 0.0}, 0
        for imgs, targets, info in dl:
            it += 1
            f = min(1.0, it / 200) * (0.1 ** sum(ep > int(m) for m in a.milestones.split(",")))
            for g in opt.param_groups:
                g["lr"] = g["base"] * f
            imgs = imgs.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outs, hs = detr_outputs(model, imgs)
            ld = crit(outs, targets)
            dloss = sum(ld[k] * crit.weight_dict[k] for k in ld if k in crit.weight_dict)
            with torch.no_grad():
                indices = crit.matcher({"pred_logits": outs["pred_logits"], "pred_boxes": outs["pred_boxes"]}, targets)
            if info["traj_labels"].shape[2] > 0:                      # clips without any object: DETR loss only
                tf, tl, tt, tm, uf, ut, um, ul = prepare_seq_info(hs, indices, info, device)
                logits = model.id_decoder(model.traj(tf), tl, tt, tm, uf, ut, um)
                iloss = id_loss(logits, ul, um)
            else:
                logits, iloss = torch.zeros(0), dloss.new_zeros(())
            loss = dloss + iloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detr_params, 0.1)
            torch.nn.utils.clip_grad_norm_(groups["other"], 0.1)
            opt.step()
            sums["detr"] += float(dloss)
            sums["id"] += float(iloss)
            n += 1
            if a.check:
                print(f"check clip {n}: frames {tuple(imgs.shape)}, ids/groups {tuple(info['traj_labels'].shape)}, "
                      f"logits {tuple(logits.shape)}, detr {float(dloss):.3f}, id {float(iloss):.3f}, "
                      f"max memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB, {time.time() - t0:.1f}s",
                      flush=True)
            elif n % 100 == 0:
                print(f"  epoch {ep} clip {n}: detr {sums['detr'] / n:.4f}  id {sums['id'] / n:.4f}  "
                      f"{(time.time() - t0) / n:.2f} s/clip", flush=True)
        print(f"Epoch {ep:02d} done  loss={(sums['detr'] + sums['id']) / max(n, 1):.4f}  detr={sums['detr'] / max(n, 1):.4f}  "
              f"id={sums['id'] / max(n, 1):.4f}  time={time.time() - t0:.1f}s", flush=True)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = {"model_state_dict": model.state_dict(), "epoch": ep, "img_size": a.img_size, "img_h": 384,
                 "detr_init": a.detr, "init": a.init, "milestones": a.milestones, "clips_per_epoch": len(ds),
                 "seed": a.seed, "check": a.check}
        torch.save(state, out.with_name("motip_check.pth" if a.check else "motip_last.pth"))
        if a.check:
            break
    if not a.check:
        torch.save(state, a.out)
        print(f"-> {a.out}  ({(time.time() - t_all) / 60:.1f} min)")


if __name__ == "__main__":
    main()
