#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of SE-UNet GHI predictor.

Prerequisites:
    1. Install dependencies: pip install -r requirements.txt
    2. Download the pretrained weights (best_model.pth) and place them
       in the same directory, or update the path below.
"""

import torch
import numpy as np
from inference import GHIPredictor, load_predictor


# ------------------------------------------------------------------ #
#  Example 1 — minimal usage with a random tensor                     #
# ------------------------------------------------------------------ #
def example_basic():
    print("=" * 50)
    print("Example 1: Basic usage")
    print("=" * 50)

    predictor = GHIPredictor(
        model_path="best_model.pth",
        device="cuda",   # set to "cpu" if no GPU is available
    )

    # Simulate a pre-processed FY-4 input (batch=4, 20 channels, 256×256 px)
    x = torch.randn(4, 20, 256, 256)
    ghi = predictor.predict(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {ghi.shape}")
    print(f"GHI range    : {ghi.min():.1f} ~ {ghi.max():.1f}  W/m²")
    print()


# ------------------------------------------------------------------ #
#  Example 2 — NumPy array input, NumPy output                        #
# ------------------------------------------------------------------ #
def example_numpy():
    print("=" * 50)
    print("Example 2: NumPy input / output")
    print("=" * 50)

    predictor = load_predictor("best_model.pth", device="cpu")

    x_np = np.random.randn(2, 20, 256, 256).astype(np.float32)
    ghi_np = predictor.predict(x_np, return_numpy=True)

    print(f"Input  (numpy): {x_np.shape}  dtype={x_np.dtype}")
    print(f"Output (numpy): {ghi_np.shape}  dtype={ghi_np.dtype}")
    print()


# ------------------------------------------------------------------ #
#  Example 3 — large dataset processed in sub-batches                 #
# ------------------------------------------------------------------ #
def example_large_batch():
    print("=" * 50)
    print("Example 3: Large-batch inference")
    print("=" * 50)

    predictor = GHIPredictor("best_model.pth")

    # 100 samples — processed 16 at a time to avoid OOM
    x = torch.randn(100, 20, 256, 256)
    ghi = predictor.predict_batch(x, batch_size=16)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {ghi.shape}")
    print()


# ------------------------------------------------------------------ #
#  Example 4 — single-sample inference (no batch dimension)           #
# ------------------------------------------------------------------ #
def example_single_sample():
    print("=" * 50)
    print("Example 4: Single-sample (no batch dim)")
    print("=" * 50)

    predictor = GHIPredictor("best_model.pth")

    x = torch.randn(20, 256, 256)        # (C, H, W) — no batch dimension
    ghi = predictor.predict(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {ghi.shape}") # (1, H, W)
    print()


if __name__ == "__main__":
    example_basic()
    example_numpy()
    example_large_batch()
    example_single_sample()