#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_NUSCENES_ROOT="$(dirname "$REPO_ROOT")/datasets/nuscenes"
NUSCENES_ROOT="${NUSCENES_ROOT:-$DEFAULT_NUSCENES_ROOT}"

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[INFO] Current python:"
which python
python --version

mkdir -p data/nuscenes_anno_pkls
mkdir -p data/nuscenes_anno_pkls/ablation
mkdir -p outputs

echo "[0/4] Check nuScenes-mini path"

if [ -L "data/nuscenes" ] && [ ! -e "data/nuscenes" ]; then
  echo "[INFO] removing broken data/nuscenes symlink"
  rm "data/nuscenes"
fi

if [ ! -e "data/nuscenes" ]; then
  if [ ! -d "$NUSCENES_ROOT" ]; then
    echo "[ERROR] nuScenes root not found: $NUSCENES_ROOT"
    echo "Set NUSCENES_ROOT to the dataset directory and retry."
    exit 1
  fi
  ln -s "$NUSCENES_ROOT" data/nuscenes
fi

if [ ! -d "data/nuscenes/v1.0-mini" ]; then
  echo "[ERROR] data/nuscenes/v1.0-mini not found."
  echo "Resolved nuScenes root: $NUSCENES_ROOT"
  exit 1
fi

echo "[OK] nuScenes-mini found."
ls data/nuscenes

echo "[1/4] Check or generate original mini pkl"

if [ ! -f "data/nuscenes_anno_pkls/nuscenes-mini_infos_train.pkl" ] || [ ! -f "data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl" ]; then
  python tools/nuscenes_converter.py \
    --root_path ./data/nuscenes \
    --version v1.0-mini \
    --info_prefix data/nuscenes_anno_pkls/nuscenes-mini
else
  echo "[OK] original mini pkl already exists."
fi

echo "[2/4] Skip anchor generation on local CPU"
echo "[INFO] anchor_generator.py requires mmcv._ext. Run it later on NVIDIA GPU server with mmcv-full."

echo "[3/4] Augment radar info for train"

python custom_tools/augment_nuscenes_radar_info.py \
  --dataroot ./data/nuscenes \
  --version v1.0-mini \
  --input-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_train.pkl \
  --output-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_train_radar.pkl

echo "[4/4] Augment radar info for val"

python custom_tools/augment_nuscenes_radar_info.py \
  --dataroot ./data/nuscenes \
  --version v1.0-mini \
  --input-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl \
  --output-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_val_radar.pkl

echo "[CHECK] generated files:"
ls -lh data/nuscenes_anno_pkls

echo "[OK] local CPU preparation without anchor finished."
