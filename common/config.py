"""
Common experiment profile and paths.

Every train / evaluate / demo script of every detector reads its settings
from here. Only the algorithm (detectors/<version>/) differs between runs.

Paths can be moved with one environment variable:
    export PHYSICAL_AI_ROOT=/home/minsoogo/Physical_AI
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_AI_ROOT = Path(os.environ.get("PHYSICAL_AI_ROOT", "~/Physical_AI")).expanduser()
DATASETS_ROOT = PHYSICAL_AI_ROOT / "datasets"

VOC_ROOT = DATASETS_ROOT / "VOC" / "VOCdevkit" / "VOC2007"
KITTI_ROOT = DATASETS_ROOT / "KITTI_5k"
DEMO_ROOT = DATASETS_ROOT / "demo"
RUNS_ROOT = FRAMEWORK_ROOT / "runs"

# ------------------------------------------------------------------
# OD-Small Benchmark Profile v1.0  (Mode A: controlled comparison)
# ------------------------------------------------------------------
PROFILE = {
    "name": "OD-Small v1.1",       # v1.1: Image training augmentation of the YOLO papers (image_aug)
    "seed": 777,
    "epochs": 10,
    "optimizer": "Adam",           # decided 2026-09-28 (legacy Complex/YOLO3D optimizer)
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_schedule": "constant",     # legacy StepLR(10) never fires within 10 epochs
    "effective_batch": 8,          # micro batch x gradient accumulation
    "hflip_prob": 0.5,             # common augmentation for every task
    # [Unified] profile v1.1 (2026-09-30): basic augmentation of the YOLOv1 / v2 papers for the Image
    # track (10 epochs from scratch with flip only memorised the training set: train cls acc 99.6 %,
    # test mAP50 < 0.04 in the 1/4-VOC convergence study). KITTI keeps its point-cloud augmentation.
    "image_aug": {"jitter": 0.2, "hue": 0.1, "saturation": 1.5, "exposure": 1.5},
    "eval_iou": 0.5,               # AP50 / mAP50
    "eval_conf": 0.01,             # low threshold -> full PR curve
    "nms_iou": 0.5,                # same NMS threshold for every task
    "pre_nms_topk": 300,
    "max_det": 100,
    "demo_conf": 0.3,
}

# Mode B (reference recipe): legacy optimizer of each project.
REFERENCE_RECIPE = {
    "image":   {"optimizer": "SGD",  "lr": 2.5e-4, "lr_schedule": "constant"},
    "complex": {"optimizer": "Adam", "lr": 1e-3,   "lr_schedule": "step10x0.8"},
    "yolo3d":  {"optimizer": "Adam", "lr": 1e-3,   "lr_schedule": "step10x0.8"},
}

# ------------------------------------------------------------------
# Task definitions (dataset side only - no algorithm here)
# ------------------------------------------------------------------
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
# [Unified] Image dataset selection (2026-10-01). Algorithm validation runs on FashionLetters36
# (data_processing_fl36.py: Fashion-MNIST 10 + EMNIST letters 26 classes, 4,096 / 1,024 images,
# VOC layout); benchmarking runs on VOC2007. Select with UNIFIED_IMAGE_DATASET=fl36 (default voc2007).
IMAGE_DATASET = os.environ.get("UNIFIED_IMAGE_DATASET", "voc2007")
FL36_CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot",
                *[chr(ord("A") + i) for i in range(26)])
FL36_ROOT = DATASETS_ROOT / "FashionLetters36"
# [Unified] MOT stage (2026-10-22): KITTI-MOTS frames converted to the same VOC layout by
# tools/mot/kittimots_to_voc.py (boxes = tight boxes of the MOTS masks, ignore regions = difficult objects).
# trainval = MOTS train (12 sequences), test = MOTS val (9 sequences). Select with UNIFIED_IMAGE_DATASET=kittimots.
KITTIMOTS_CLASSES = ("car", "pedestrian")
KITTIMOTS_VOC_ROOT = DATASETS_ROOT / "KITTI_MOTS_VOC"
if IMAGE_DATASET == "fl36":
    VOC_ROOT, VOC_CLASSES = FL36_ROOT, FL36_CLASSES
elif IMAGE_DATASET == "kittimots":
    VOC_ROOT, VOC_CLASSES = KITTIMOTS_VOC_ROOT, KITTIMOTS_CLASSES
elif IMAGE_DATASET != "voc2007":
    raise ValueError(f"UNIFIED_IMAGE_DATASET must be voc2007, fl36 or kittimots, not {IMAGE_DATASET!r}")

KITTI_CLASSES = ("Car", "Pedestrian", "Cyclist")

# [Unified] Segmentation stage (2026-10-11). Instance segmentation data in VOC layout:
#   JPEGImages/, Annotations/ (boxes, unchanged), SegmentationObject/ (instance id PNG: k = k-th XML object,
#   0 = background, 255 = void), SegmentationClass/ (class id + 1 PNG), ImageSets/Segmentation/<split>.txt.
# FL36-Seg (default) is FashionLetters36 regenerated with the same seed plus masks; VOC2007 is the check set
# (422 trainval / 210 test images with masks). Select with UNIFIED_SEG_DATASET=fl36seg | voc2007.
SEG_DATASET = os.environ.get("UNIFIED_SEG_DATASET", "fl36seg")
FL36SEG_ROOT = DATASETS_ROOT / "FashionLetters36_Seg"
if SEG_DATASET == "fl36seg":
    SEG_ROOT, SEG_CLASSES = FL36SEG_ROOT, FL36_CLASSES
elif SEG_DATASET == "voc2007":
    SEG_ROOT = DATASETS_ROOT / "VOC" / "VOCdevkit" / "VOC2007"
    SEG_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor")
elif SEG_DATASET == "toy":                        # tools/make_toy_seg.py (pipeline smoke test only)
    SEG_ROOT, SEG_CLASSES = DATASETS_ROOT / "ToySeg", ("square", "disc", "triangle")
else:
    raise ValueError(f"UNIFIED_SEG_DATASET must be fl36seg, voc2007 or toy, not {SEG_DATASET!r}")

TASKS = {
    "image": {
        "dataset": IMAGE_DATASET,
        "classes": VOC_CLASSES,
        "img_size": 608,             # decided 2026-09-28 (same as the BEV map)
        "micro_batch": 2,            # x accum 4 = effective 8 (fits 8 GB)
        "train_split": "trainval",   # 5,011 images
        "test_split": "test",        # 4,952 images
        "demo_image": FL36_ROOT / "JPEGImages" / "test_00000.jpg" if IMAGE_DATASET == "fl36"
                      else DEMO_ROOT / "images" / "dog.jpg",
        "demo_video": DEMO_ROOT / "videos" / "Highway.mp4",
    },
    "complex": {
        "dataset": "kitti5k",
        "classes": KITTI_CLASSES,
        "img_size": 608,             # BEV map size (fixed by the BEV grid)
        "micro_batch": 2,
        "train_split": "trainval",   # 5,011 frames
        "test_split": "test",        # 2,470 frames
    },
    "seg": {
        "dataset": SEG_DATASET,
        "classes": SEG_CLASSES,
        "img_size": 608,
        "micro_batch": 2,
        "train_split": "trainval",
        "test_split": "test",
        "mask_threshold": 0.5,       # mask probability -> binary
        "demo_image": SEG_ROOT / "JPEGImages" / "test_00000.jpg",
        "demo_video": DEMO_ROOT / "videos" / "Highway.mp4",
    },
    "semseg": {                      # [Unified] 2026-10-11 semantic segmentation on the same SEG dataset
        "dataset": SEG_DATASET,
        "classes": SEG_CLASSES,      # + background (index 0) inside the models: K = C + 1
        "img_size": 608,
        "micro_batch": 2,
        "train_split": "trainval",
        "test_split": "test",
        "demo_image": SEG_ROOT / "JPEGImages" / "test_00000.jpg",
        "demo_video": DEMO_ROOT / "videos" / "Highway.mp4",
    },
    "yolo3d": {
        "dataset": "kitti5k",
        "classes": KITTI_CLASSES,
        "img_size": 608,
        "micro_batch": 2,
        "train_split": "trainval",
        "test_split": "test",
    },
}


def run_dir(task, version):
    return RUNS_ROOT / f"{task}_{version}"


def checkpoint_path(task, version, epochs):
    return run_dir(task, version) / "checkpoints" / f"{task}_{version}_{TASKS[task]['dataset']}_e{epochs:03d}.pth"
