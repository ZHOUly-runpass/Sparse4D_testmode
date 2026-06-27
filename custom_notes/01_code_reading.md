# Sparse4D 本地开发记录

## 当前硬件状态

本地电脑没有 NVIDIA GPU，因此本地不进行 Sparse4D CUDA op 编译和模型训练。

## 本地负责

- Sparse4D 代码阅读
- nuScenes-mini 数据准备
- pkl / anchor 生成
- camera / radar 模态剥离方案设计
- radar 信息增强脚本开发
- radar loader 开发
- 消融实验 pkl 生成
- 后续上传服务器训练

## 服务器负责

- CUDA 环境配置
- mmcv-full CUDA 版本安装
- Sparse4D 自定义 CUDA op 编译
- Sparse4Dv3 mini debug 训练
- camera / radar 融合模型训练与评测

## 重点文件

- projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
- tools/nuscenes_converter.py
- tools/anchor_generator.py
- projects/mmdet3d_plugin/models/
- projects/mmdet3d_plugin/ops/

## 当前本地执行入口（2026-06-27）

当前 WSL 项目路径：

```text
/mnt/d/E2Eproject_Sparse4D/Sparse4D
```

nuScenes-mini 默认路径：

```text
/mnt/d/E2Eproject_Sparse4D/datasets/nuscenes
```

如果数据位于其他目录，运行准备脚本前设置 `NUSCENES_ROOT`。本地 CPU
环境使用不生成 anchor 的入口：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sparse4d_cpu
cd /mnt/d/E2Eproject_Sparse4D/Sparse4D
bash scripts/prepare_local_cpu_no_anchor.sh
python custom_tools/make_modality_ablation_pkls.py
python custom_tools/test_radar_loader.py
python custom_tools/test_radar_loader.py \
  --pkl data/nuscenes_anno_pkls/nuscenes-mini_infos_val_radar.pkl
python custom_tools/visualize_radar_bev.py
```

radar loader 当前输出 lidar 坐标系下的 `[N, 7]`：
`x, y, z, vx_comp, vy_comp, rcs, channel_id`。位置和补偿速度均会从
radar sensor 坐标系转换到 ego，再转换到 lidar 坐标系。

`sparse4dv3_mini_cam_radar.py` 和
`sparse4dv3_mini_cam_radar_dropout.py` 当前仅完成 radar PKL、loader、
格式化和 Collect 的 pipeline smoke-test 接线。Sparse4D 模型尚无 radar
encoder/fusion branch，因此不能把这些配置视为已完成的融合训练配置。
