"""
GHI Inference Interface for SE-UNet

Usage:
    from inference import GHIPredictor
    predictor = GHIPredictor("best_model.pth", device="cuda")
    ghi = predictor.predict(satellite_data)  # shape: (B, 1, H, W), unit: W/m²
"""

import torch
import numpy as np
from typing import Optional, Union
from model import ImprovedUNet


class GHIPredictor:
    """
    GHI (Global Horizontal Irradiance) predictor from FY-4 satellite data.

    Input format (B, 20, H, W):
        Channel  0-12 : 13 spectral bands, normalized (visible / 0.15, thermal / 350.0)
        Channel 13    : sin(SatelliteAzimuth)
        Channel 14    : cos(SatelliteAzimuth)
        Channel 15    : SatelliteZenith / 90
        Channel 16    : sin(SunAzimuth)
        Channel 17    : cos(SunAzimuth)
        Channel 18    : SunGlintAngle / 180
        Channel 19    : SunZenith / 90

    Output:
        (B, 1, H, W)  GHI in W/m²  (approximate range 0–1300)

    Constraints:
        H and W must be multiples of 16.

    Example:
        >>> predictor = GHIPredictor("best_model.pth")
        >>> x = torch.randn(4, 20, 256, 256)
        >>> ghi = predictor.predict(x)
        >>> print(ghi.shape)      # torch.Size([4, 1, 256, 256])
        >>> print(ghi.min().item(), ghi.max().item())
    """

    def __init__(
            self,
            model_path: str,
            device: Optional[str] = None,
            in_channels: int = 20,
            model_ch: int = 64,
            ghi_scale: float = 1000.0,
    ):
        """
        Args:
            model_path  : Path to the trained model weights (.pth file).
            device      : 'cuda' or 'cpu'. Auto-detected when None.
            in_channels : Number of input channels (default 20).
            model_ch    : Base channel width used during training (default 64).
            ghi_scale   : De-normalisation factor applied to raw model output.
                          Must match the y_scale used during training (default 1000.0).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.ghi_scale = ghi_scale

        self.model = ImprovedUNet(
            in_channels=in_channels,
            out_channels=1,
            ch=model_ch,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"[GHIPredictor] device : {self.device}")
        print(f"[GHIPredictor] weights: {model_path}")

    def predict(
            self,
            x: Union[torch.Tensor, np.ndarray],
            return_numpy: bool = False,
            apply_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Run inference on pre-processed satellite data.

        Args:
            x           : Input tensor/array of shape (B, 20, H, W) or (20, H, W).
            return_numpy: If True, return np.ndarray; otherwise return torch.Tensor.
            apply_mask  : Optional binary mask (H, W) or (1, 1, H, W).
                          Pixels where mask == 0 are set to 0 in the output.

        Returns:
            GHI prediction in W/m², shape (B, 1, H, W) or (1, H, W).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        with torch.no_grad():
            pred = self.model(x) * self.ghi_scale

        if apply_mask is not None:
            if apply_mask.dim() == 2:
                apply_mask = apply_mask.unsqueeze(0).unsqueeze(0)
            pred = pred * apply_mask.to(self.device)

        if squeeze:
            pred = pred.squeeze(0)

        return pred.cpu().numpy() if return_numpy else pred.cpu()

    def predict_batch(
            self,
            x: Union[torch.Tensor, np.ndarray],
            batch_size: int = 16,
            return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process a large tensor in smaller chunks to avoid OOM errors.

        Args:
            x          : Input of shape (N, 20, H, W).
            batch_size : Number of samples processed per forward pass.
            return_numpy: If True, return np.ndarray.

        Returns:
            GHI predictions of shape (N, 1, H, W).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        results = [
            self.predict(x[i: i + batch_size], return_numpy=False)
            for i in range(0, x.shape[0], batch_size)
        ]
        out = torch.cat(results, dim=0)
        return out.numpy() if return_numpy else out


def load_predictor(model_path: str, device: Optional[str] = None) -> GHIPredictor:
    """Convenience wrapper around GHIPredictor."""
    return GHIPredictor(model_path, device)