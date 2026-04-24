# SE-UNet for GHI Estimation from FY-4 Satellite Data

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