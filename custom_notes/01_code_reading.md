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
