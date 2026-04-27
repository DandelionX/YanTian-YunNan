# SE-UNet 太阳辐射反演模型

基于 FY-4A 静止卫星数据的全球水平辐照度（GHI）估算模型，采用 SE-UNet 架构，支持 4km 分辨率下的逐小时太阳辐射反演。

---

## 📋 目录

- [项目简介](#项目简介)
- [模型功能](#模型功能)
- [目录结构](#目录结构)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [模型性能](#模型性能)
- [模型架构](#模型架构)
- [作者信息](#作者信息)

---

## 项目简介

本项目实现了基于 FY-4A 静止卫星 L1 级多通道数据的太阳辐射反演模型，利用深度学习方法估算地表全球水平辐照度（GHI，Global Horizontal Irradiance）。

- **输入**：FY-4A 卫星 13 个光谱通道 + 5 个几何角度参数，共 20 通道，空间分辨率 4km
- **输出**：逐像元 GHI 预测值（单位：W/m²），空间分辨率 4km
- **本仓库发布推理部分代码**，训练代码与数据集暂不予开放，有需要可联系作者

---

## 模型功能

| 功能 | 说明 |
|------|------|
| 单样本推理 | 输入单张卫星图像，输出 GHI 空间分布图 |
| 批量推理 | 支持大批量数据分批处理，避免显存溢出 |
| NumPy 接口 | 支持 numpy 数组直接输入/输出，便于与其他工具链集成 |
| 可选掩码 | 支持传入陆地/海洋掩码，屏蔽无效区域 |
| 自动设备选择 | 自动检测 CUDA，无 GPU 时回退至 CPU |

---

## 目录结构

```
se-unet-solar-ghi/
├── model.py          # SE-UNet 模型结构（仅推理）
├── inference.py      # GHIPredictor 推理接口封装
├── example.py        # 使用示例
├── requirements.txt  # 依赖列表
└── README.md
```

---

## 数据准备

### 输入格式

模型输入为预处理后的 20 通道张量，形状 `(B, 20, H, W)`，H 和 W 必须为 16 的倍数（推荐 256×256）。

| 通道索引 | 内容 | 归一化方式 |
|----------|------|-----------|
| 0–5 | FY-4A 可见光 / 近红外通道（Ch01–Ch06） | ÷ 0.15 |
| 6–12 | FY-4A 热红外 / 中红外通道（Ch08–Ch14） | ÷ 350.0 |
| 13 | sin(卫星方位角) | — |
| 14 | cos(卫星方位角) | — |
| 15 | 卫星天顶角 | ÷ 90 |
| 16 | sin(太阳方位角) | — |
| 17 | cos(太阳方位角) | — |
| 18 | 太阳耀光角 | ÷ 180 |
| 19 | 太阳天顶角 | ÷ 90 |

> **注**：FY-4A Ch07（3.75μm 中红外）因训练数据中该通道存在缺失，未纳入输入特征，通道索引 6–12 对应原始 Ch08–Ch14。

### 输出格式

| 项目 | 说明 |
|------|------|
| 形状 | `(B, 1, H, W)` |
| 单位 | W/m² |
| 值域 | 约 0–1000 W/m²（夜间接近 0） |

---

## 使用方法

### 快速开始

```python
from inference import GHIPredictor
import torch

# 加载模型
predictor = GHIPredictor(
    model_path="best_model.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 输入：预处理后的 FY-4 数据，形状 (B, 20, H, W)
x = torch.randn(4, 20, 256, 256)
ghi = predictor.predict(x)   # 输出 (B, 1, H, W)，单位 W/m²

print(ghi.shape)             # torch.Size([4, 1, 256, 256])
```

### NumPy 输入 / 输出

```python
import numpy as np
from inference import load_predictor

predictor = load_predictor("best_model.pth", device="cpu")

x_np = np.random.randn(2, 20, 256, 256).astype(np.float32)
ghi_np = predictor.predict(x_np, return_numpy=True)  # 返回 np.ndarray
```

### 大批量分批推理

```python
# 100 个样本，每次处理 16 个，避免显存不足
x = torch.randn(100, 20, 256, 256)
ghi = predictor.predict_batch(x, batch_size=16)
```

更多用法见 [`example.py`](example.py)。

### 预训练权重

| 链接 | 说明 |
|------|------|
| 如需权重文件，请联系作者 | 基于 FY-4A 2021年1月中国区域数据训练 |

---

## 模型性能

在独立测试集上的评估结果（单位：W/m²）：

| 区域 | MAE | RMSE | R | R² | SSIM | PSNR |
|------|-----|------|---|----|------|------|
| 云南 | 35.40 | 49.76 | 0.9630 | 0.9241 | 0.8875 | 32.08 dB |
| 全国 | 18.93 | 28.88 | 0.9928 | 0.9854 | 0.9353 | 31.17 dB |


---

## 模型架构

SE-UNet：在标准 UNet 的瓶颈层及每个解码器阶段插入 **Squeeze-and-Excitation (SE) 注意力模块**，增强通道间特征选择能力。

```
输入 (B, 20, H, W)
    │
Conv2d stem ────────────────────────────────── skip4
    │
AvgPool + Conv ──────────────────────────── skip3
    │
AvgPool + Conv ──────────────────────── skip2
    │
AvgPool + Conv ──────────────────── skip1
    │
AvgPool + Conv
    │
  [SE 瓶颈]
    │
Upsample + Conv ← cat(skip1) ← [SE]
    │
Upsample + Conv ← cat(skip2) ← [SE]
    │
Upsample + Conv ← cat(skip3) ← [SE]
    │
Upsample + Conv ← cat(skip4) ← [SE]
    │
Conv2d 1×1 + Sigmoid
    │
输出 (B, 1, H, W)  ×1000 → GHI（W/m²）
```

---



## 作者信息

模型训练:**陈颖**（中国地质大学（北京））、侯宇田（哈尔滨工业大学）等，中国科学院大气物理研究所（IAP/CAS），RUMLA。
<img width="344" height="344" alt="8d8101fcf37782ab5120fdb51ce797ad" src="https://github.com/user-attachments/assets/a9f16a79-867d-483a-894c-a4319bd12a5f" />

如有问题或需要预训练权重，欢迎通过 GitHub Issues 联系。

---

## 许可证

[MIT License](LICENSE)# SE-UNet for GHI Estimation from FY-4 Satellite Data

A PyTorch implementation of an SE-UNet model that estimates **Global Horizontal Irradiance (GHI)** from FY-4 geostationary satellite imagery.

---

## Architecture

The model is a **UNet** with **Squeeze-and-Excitation (SE) attention blocks** inserted at the bottleneck and every decoder stage.

```
Input (B, 20, H, W)
    │
    ▼
Conv2d stem  ──────────────────────────────────────────────────────┐ skip4
    │                                                              │
AvgPool + Conv  ────────────────────────────────────────────── skip3│
    │                                                          │   │
AvgPool + Conv  ────────────────────────────────────────── skip2│   │
    │                                                      │   │   │
AvgPool + Conv  ────────────────────────────────────── skip1│   │   │
    │                                                  │   │   │   │
AvgPool + Conv                                         │   │   │   │
    │                                                  │   │   │   │
[SE Bottleneck]                                        │   │   │   │
    │                                                  │   │   │   │
Upsample + Conv ← cat(skip1) ← [SE] ──────────────────┘   │   │   │
    │                                                      │   │   │
Upsample + Conv ← cat(skip2) ← [SE] ──────────────────────┘   │   │
    │                                                          │   │
Upsample + Conv ← cat(skip3) ← [SE] ──────────────────────────┘   │
    │                                                              │
Upsample + Conv ← cat(skip4) ← [SE] ──────────────────────────────┘
    │
Conv2d 1×1 head
    │
Output (B, 1, H, W)   ×1000 → GHI in W/m²
```

---

## Input Specification

| Index | Content | Normalisation |
|-------|---------|--------------|
| 0–5   | Visible / near-IR spectral bands (Ch01–Ch06) | ÷ 0.15 |
| 6–12  | Thermal / mid-IR spectral bands (Ch08–Ch14) | ÷ 350.0 |
| 13    | sin(SatelliteAzimuth) | — |
| 14    | cos(SatelliteAzimuth) | — |
| 15    | SatelliteZenith | ÷ 90 |
| 16    | sin(SunAzimuth) | — |
| 17    | cos(SunAzimuth) | — |
| 18    | SunGlintAngle | ÷ 180 |
| 19    | SunZenith | ÷ 90 |

> **Spatial constraints:** H and W must both be multiples of 16.  
> Recommended patch size: 256 × 256.

---

## Installation

```bash
git clone https://github.com/chenxiaoxiao-cy/se-unet-ghi.git
cd se-unet-ghi
pip install -r requirements.txt
```

Requires Python ≥ 3.8 and PyTorch ≥ 2.0.

---

## Quick Start

```python
from inference import GHIPredictor
import torch

predictor = GHIPredictor(
    model_path="best_model.pth",
    device="cuda",      # or "cpu"
)

# x shape: (B, 20, H, W) — pre-processed FY-4 data
x = torch.randn(4, 20, 256, 256)
ghi = predictor.predict(x)          # → (B, 1, H, W), unit: W/m²

print(ghi.shape)                    # torch.Size([4, 1, 256, 256])
```

More examples (NumPy input, large-batch processing, single-sample) are in [`example.py`](example.py).

---

## Pretrained Weights

Download `best_model.pth` and place it in the project root:

| Link | Description |
|------|-------------|
|Weights available upon request| Trained on FY-4A data over China, January 2021 |

> **Note:** Contact the author if you need the pretrained weights.

---

## File Structure

```
se-unet-ghi/
├── model.py          # SE-UNet architecture (inference-only)
├── inference.py      # GHIPredictor class
├── example.py        # Usage examples
├── requirements.txt
└── README.md
```

Training code and data-preprocessing pipelines are **not included** in this release, as the training dataset is proprietary.

---

## Citation

If you use this code or the pretrained weights in your work, please cite this repository:

```bibtex
@software{se_unet_ghi,
  author  = {Chen Ying},
  title   = {SE-UNet for GHI Estimation from FY-4 Satellite Data},
  year    = {2025},
  url     = {https://github.com/chenxiaoxiao-cy/se-unet-ghi}
}
```

---

## License

[MIT License](LICENSE)
