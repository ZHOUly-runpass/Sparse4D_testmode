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

echo "[0/4] Check nuScenes-mini path"

if [ ! -d "data/nuscenes/v1.0-mini" ]; then
  echo "[ERROR] data/nuscenes/v1.0-mini not found."
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
