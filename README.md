# MOT_E2E_MOTIP

**MOTIP** - end-to-end tracking on KITTI-MOTS, trained from scratch.

- Paper: Gao et al., CVPR 2025; official code: `MCG-NJU/MOTIP`
- Implementation: `common/mot/motip.py` (the module docstring below lists what was checked against the official
  code and every `[Unified]` difference)

## Results (KITTI-MOTS val)

| run | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | ms / frame |
|---|---|---|---|---|---|---|---|
| motip | 0.305 | 0.419 | 0.222 | 0.277 | 0.279 | 2015 | 72.3 |
| motip_e13 | 0.366 | 0.440 | 0.306 | 0.375 | 0.390 | 1151 | 72.2 |
| motip_e25 | 0.409 | 0.461 | 0.366 | 0.433 | 0.436 | 598 | 72.3 |
| motip_e37 | 0.408 | 0.449 | 0.375 | 0.424 | 0.429 | 444 | 72.2 |
| motip_e49 | 0.434 | 0.471 | 0.402 | 0.446 | 0.464 | 401 | 72.7 |
| motip_e61 | 0.419 | 0.480 | 0.369 | 0.460 | 0.449 | 384 | 72.3 |

The table reports the last epoch (61) as the protocol value; the epoch rows are the training curve (5 -> 61 epochs), selecting the best epoch would use the test set.
Comparison with tracking-by-detection on the same network: `RESULTS.md` of `MOT_Detectors_KITTIMOTS`, section 4.

## Implementation notes

```
MOT stage (2026-10-24): MOTIP (Gao, Zhang, Wang, "Multiple Object Tracking as ID Prediction", CVPR 2025), E2E.
Official code MCG-NJU/MOTIP (main, ffc0e90), checked file by file:
  models/motip/id_decoder.py          6 layers; layer 0 cross-attention only, layers 1-5 self-attention among the
                                      current detections then cross-attention to the trajectories; features 256 +
                                      ID embedding 256 (one-hot of 50 ID words + 1 "newborn" word -> linear), heads of
                                      32 dims, add_zero_attn, relative temporal position bias (30 steps, per head),
                                      causal mask (a trajectory token is visible only if its time < current time)
  models/motip/trajectory_modeling.py adapter FFN + LN + FFN + LN on the DETR decoder output (GELU, ratio 2)
  data/transforms.py                  GenerateIDLabels: per clip and per group (6 groups) a random permutation of the
                                      50 ID words; TurnIntoTrajectoryAndUnknown: trajectory occlusion (p 0.5, a random
                                      span masked), trajectory switch (p 0.5 per object per frame, shuffled among the
                                      selected), first appearance -> "newborn" label
  models/motip/id_criterion.py        cross entropy on the current detections, first frame excluded
  train.py                            clip of 30 frames (random interval 1-4), 4 random frames with gradient through
                                      DETR, the rest without; DETR set loss on all frames; the matched query outputs
                                      feed the ID part; AdamW 1e-4 (backbone x0.1, sampling_offsets /
                                      reference_points x0.05), weight decay 5e-4, grad-norm clip 0.1 separately for DETR
                                      and the rest, 10 epochs (x0.1 at 6 and 9), 1 warm-up epoch, batch 1
  models/runtime_tracker.py           det_thresh 0.3, newborn_thresh 0.6, id_thresh 0.2, object-max assignment,
                                      trajectory buffer of the last 30 frames (miss_tolerance 30)

[Unified] decisions:
  * detection part = our KITTI-MOTS Deformable DETR (detectors/deformable_detr, 30 epochs, 1216 x 384, no box
    refinement, 2 classes, RGB / 255) - the same detector as the "deformable_detr" row of the tracking table, so
    MOTIP is compared with tracking-by-detection on identical detection weights (official: COCO + dataset
    pre-trained Deformable DETR with box refinement, ImageNet normalisation).
  * BatchNorm of the detector frozen during MOTIP training (batch 1 clip); the official detector uses
    FrozenBatchNorm2d in the backbone.
  * at most 30 identities per training clip enter the ID part (official NUM_TRAINING_IDS; all objects still train
    the detector), ID decoder layers with activation checkpointing (official USE_DECODER_CHECKPOINT option).
  * reduced schedule for 8 GB / the GPU window: 5 epochs of 1000 random clips (official: every clip start, 10
    epochs), steps x0.1 at epochs 3 and 4, linear warm-up over the first 200 iterations; DETR frames with gradient
    run one at a time with activation checkpointing; common augmentation, identical for all frames of a clip.
```

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

## Usage

```bash
pip install -r requirements.txt
pip install "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/train_motip.py --check     # smoke test
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/train_motip.py             # training
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/run_motip.py --weights runs/motip/motip_kittimots_e061.pth --name motip_e61 --causal
```

Trained weights: the GitHub Release of this repository (`motip_kittimots_e061.pth`, put it under `runs/motip/`).

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
