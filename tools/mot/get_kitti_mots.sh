#!/usr/bin/env bash
# MOT stage (2026-10-22): download KITTI tracking left images + KITTI-MOTS instance PNGs, unpack only what we use.
#   bash tools/mot/get_kitti_mots.sh            # resumable (wget -c); re-run after a dropped connection
# Result:
#   ~/Physical_AI/datasets/KITTI_MOTS/training/image_02/<seq>/<frame>.png   (21 training sequences only)
#   ~/Physical_AI/datasets/KITTI_MOTS/instances/<seq>/<frame>.png          (16-bit: class*1000+id, 10000 = ignore)
# Writes ~/Physical_AI/datasets/KITTI_MOTS/DOWNLOAD_OK when both parts are in place (the GPU runner waits for it).
# Licence: KITTI-MOTS annotations are CC BY-NC-SA 3.0 -> weights trained on them carry the NC condition.
set -u
R=$HOME/Physical_AI/datasets; DL=$R/downloads; OUT=$R/KITTI_MOTS
mkdir -p "$DL" "$OUT"
log() { echo "$(date '+%F %T')  $*"; }

IMG_URL=https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
STEP_URL=http://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz
MOTS_URLS="https://www.vision.rwth-aachen.de/media/resource_files/instances.zip
https://omnomnom.vision.rwth-aachen.de/data/mots/instances.zip"

fetch() {  # fetch <url> <file>: HEAD check, then resumable download
  local url=$1 f=$2
  if ! curl -sfIL --max-time 30 "$url" > /dev/null; then return 1; fi
  wget -c -q --show-progress -O "$f" "$url"
}

log "free disk: $(df -h "$R" | tail -1 | awk '{print $4}')  (need ~25 GB: 15 GB zip + ~9 GB unpacked training images)"

# 1. KITTI tracking left colour images (~15 GB). Only training/image_02 is unpacked (test sequences not used).
if [ ! -d "$OUT/training/image_02/0020" ]; then
  fetch "$IMG_URL" "$DL/data_tracking_image_2.zip" || { log "FAILED: $IMG_URL  -> download it from cvlibs.net (tracking > left color images) into $DL and re-run"; exit 1; }
  log "unzip training/image_02 ..."
  unzip -q -n "$DL/data_tracking_image_2.zip" "training/image_02/*" -d "$OUT" || { log "unzip failed (truncated zip? delete it and re-run)"; exit 1; }
fi
log "image sequences: $(ls "$OUT/training/image_02" | wc -l) (expect 21)"

# 2. KITTI-MOTS instance annotations (PNG format, train+val)
if [ ! -d "$OUT/instances/0020" ]; then
  ok=0
  for u in $MOTS_URLS; do
    fetch "$u" "$DL/kitti_mots_instances.zip" && { ok=1; break; }
    log "not reachable: $u"
  done
  if [ $ok = 1 ]; then
    unzip -q -n "$DL/kitti_mots_instances.zip" -d "$OUT/_tmp_instances"
    d0=$(find "$OUT/_tmp_instances" -type d -name 0000 | head -1)
    [ -n "$d0" ] && mv "$(dirname "$d0")" "$OUT/instances" && rm -rf "$OUT/_tmp_instances"
  else
    # 10-22: vision.rwth-aachen.de refused connections -> KITTI-STEP panoptic maps (same sequences and split,
    # car / person instances merged from KITTI-MOTS), converted to the MOTS instance PNG encoding.
    log "MOTS site not reachable -> KITTI-STEP"
    wget -c -q -O "$DL/kitti-step.tar.gz" "$STEP_URL" || { log "FAILED: $STEP_URL"; exit 1; }
    tar -xzf "$DL/kitti-step.tar.gz" -C "$R"
    python "$(dirname "$0")/step_to_mots_png.py" --step "$R/kitti-step" --out "$OUT/instances" || exit 1
  fi
fi
log "annotation sequences: $(ls "$OUT/instances" | wc -l) (expect 21)"

n_img=$(ls "$OUT/training/image_02" | wc -l); n_ann=$(ls "$OUT/instances" | wc -l)
if [ "$n_img" = 21 ] && [ "$n_ann" = 21 ]; then
  date '+%F %T' > "$OUT/DOWNLOAD_OK"; log "DOWNLOAD_OK"
else
  log "incomplete: images $n_img / annotations $n_ann"; exit 1
fi
