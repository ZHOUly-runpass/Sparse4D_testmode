#!/usr/bin/env bash
set -e

cd /mnt/d/e2eproject/Sparse4D
export PYTHONPATH=$PWD:$PYTHONPATH

echo "[INFO] Current python:"
which python
python --version

mkdir -p data/nuscenes_anno_pkls
mkdir -p data/nuscenes_anno_pkls/ablation
mkdir -p outputs

echo "[0/5] Check nuScenes-mini path"

if [ ! -d "data/nuscenes" ]; then
  echo "[INFO] data/nuscenes not found, creating symlink..."
  ln -s /mnt/d/e2eproject/datasets/nuscenes data/nuscenes
fi

if [ ! -d "data/nuscenes/v1.0-mini" ]; then
  echo "[ERROR] data/nuscenes/v1.0-mini not found."
  echo "Expected path: /mnt/d/e2eproject/datasets/nuscenes/v1.0-mini"
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
