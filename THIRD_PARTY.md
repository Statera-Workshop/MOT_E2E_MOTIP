# Third-party code

The trackers and networks are re-implemented after the official code named below (procedure, defaults); differences
are marked `[Unified]`. SORT and DeepSORT are published under GPL-3.0 by their authors; their procedures are
re-implemented here independently, no code was copied. Data: KITTI / KITTI-MOTS / KITTI-STEP, CC BY-NC-SA 3.0. This
list is informational, not legal advice.

| source | licence (as published by the source) |
| --- | --- |
| JonathonLuiten/TrackEval (installed with pip, pinned commit 12c8791) | MIT |
| rlabbe/filterpy Kalman filter equations (re-implemented) | MIT |
| bochinski/iou-tracker (procedure re-implemented) | MIT |
| abewley/sort (procedure re-implemented independently; no code copied) | GPL-3.0 (original) |
| ifzhang/ByteTrack (procedure and defaults re-implemented) | MIT |
| noahcao/OC_SORT (procedure and defaults re-implemented) | MIT |
| ymzis69/HybridSORT (procedure and defaults re-implemented) | MIT |
| NirAharon/BoT-SORT (procedure and defaults re-implemented) | MIT |
| nwojke/deep_sort (procedure re-implemented independently; no code copied) | GPL-3.0 (original) |
| kamkyu94/TrackTrack (procedure and defaults re-implemented) | MIT |
| ifzhang/FairMOT (network heads, loss, association re-implemented) | MIT |
| MCG-NJU/MOTIP (ID decoder, ID labels, runtime tracker re-implemented) | Apache-2.0 |
| fundamentalvision/Deformable-DETR (official code) | Apache-2.0 |
| facebookresearch/detr (official code) | Apache-2.0 |
| pytorch/vision (ResNet, SSD, DCN ops) | BSD-3-Clause |
