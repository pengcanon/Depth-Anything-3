"""
depth_map.py
------------
Extract depth maps for one or more image frames using Depth-Anything-3.

Public API
----------
get_depth_maps(image_paths, model=None, device=None)
    -> prediction  (depth_anything_3.specs.Prediction)
       prediction.depth            : np.ndarray [N, H, W]  float32
       prediction.processed_images : np.ndarray [N, H, W, 3] uint8
"""

from __future__ import annotations
import torch
from depth_anything_3.api import DepthAnything3

_DEFAULT_MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE"
_model_cache: dict = {}


def load_model(model_name: str = _DEFAULT_MODEL_NAME, device: torch.device | None = None):
    """Load (or return cached) DA3 model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, str(device))
    if key not in _model_cache:
        print(f"[depth_map] Loading model {model_name} on {device} ...")
        model = DepthAnything3.from_pretrained(model_name).to(device)
        _model_cache[key] = model
    return _model_cache[key]


def get_depth_maps(
    image_paths: list[str],
    model=None,
    device: torch.device | None = None,
):
    """
    Run DA3 inference on a list of image file paths.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to input images.
    model : DepthAnything3 | None
        Pre-loaded model; loaded automatically if None.
    device : torch.device | None
        Target device; auto-detected if None.

    Returns
    -------
    prediction : depth_anything_3.specs.Prediction
        .depth            [N, H, W]   float32 – per-pixel depth
        .processed_images [N, H, W, 3] uint8  – images resized to model resolution
    """
    if model is None:
        model = load_model(device=device)

    prediction = model.inference(image_paths)
    return prediction
