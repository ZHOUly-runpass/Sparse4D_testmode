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

echo "[0/5] Check nuScenes-mini path"

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

echo "[1/5] Generate original mini pkl"
python tools/nuscenes_converter.py \
  --root_path ./data/nuscenes \
  --version v1.0-mini \
  --info_prefix data/nuscenes_anno_pkls/nuscenes-mini

echo "[2/5] Generate anchor"
python tools/anchor_generator.py \
  --ann_file data/nuscenes_anno_pkls/nuscenes-mini_infos_train.pkl \
  --output_file_name nuscenes_kmeans900.npy

echo "[3/5] Augment radar info for train"
python custom_tools/augment_nuscenes_radar_info.py \
  --dataroot ./data/nuscenes \
  --version v1.0-mini \
  --input-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_train.pkl \
  --output-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_train_radar.pkl

echo "[4/5] Augment radar info for val"
python custom_tools/augment_nuscenes_radar_info.py \
  --dataroot ./data/nuscenes \
  --version v1.0-mini \
  --input-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl \
  --output-pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_val_radar.pkl

echo "[5/5] Check generated files"
ls -lh data/nuscenes_anno_pkls
ls -lh nuscenes_kmeans900.npy

echo "[OK] local CPU preparation finished."
