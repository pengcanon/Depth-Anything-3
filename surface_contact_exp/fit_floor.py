"""
fit_floor.py
------------
1. Segment the floor region from an image frame using SAM
   (prompted with a point at the bottom-centre of the frame).
2. Extract the corresponding depth values from a DA3 depth map.
3. Fit an inverse-depth plane:  alpha*x + beta*y + gamma = 1/d

Public API
----------
load_sam(checkpoint, device=None) -> SamPredictor
segment_floor(img_rgb, predictor) -> floor_mask (H, W) bool
fit_floor_plane(floor_mask, depth_map) -> (alpha, beta, gamma) | None
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
MIN_FLOOR_PIXELS = 10


def load_sam(
    checkpoint: str = SAM_CHECKPOINT,
    device: torch.device | None = None,
) -> SamPredictor:
    """Load SAM ViT-H and return a SamPredictor ready for use."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fit_floor] Loading SAM ({checkpoint}) on {device} ...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def segment_floor(
    img_rgb: np.ndarray,
    predictor: SamPredictor,
) -> np.ndarray | None:
    """
    Segment the floor using a single SAM point prompt at
    (width//2, 4*height//5) – the bottom-centre region.

    Parameters
    ----------
    img_rgb : np.ndarray  [H, W, 3] uint8
    predictor : SamPredictor

    Returns
    -------
    floor_mask : np.ndarray [H, W] bool  or  None if segmentation fails
    """
    h, w = img_rgb.shape[:2]
    prompt_point = np.array([[w // 2, int(5 * h / 6)]])
    prompt_label = np.array([1])  # 1 = foreground

    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=prompt_point,
        point_labels=prompt_label,
        multimask_output=False,
    )

    if masks is not None and len(masks) > 0:
        return masks[0].astype(bool)
    return None


def fit_floor_plane(
    floor_mask: np.ndarray,
    depth_map: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    Fit an inverse-depth plane to the floor pixels.

    The plane model is:  alpha*x + beta*y + gamma = 1/d
    where (x, y) are pixel coordinates in depth_map space.

    Parameters
    ----------
    floor_mask : np.ndarray [Hm, Wm] bool
        Floor segmentation mask (original image resolution).
    depth_map  : np.ndarray [Hd, Wd] float32
        DA3 depth values (model output resolution).

    Returns
    -------
    (alpha, beta, gamma) : tuple of float  or  None if fit fails
    """
    dh, dw = depth_map.shape[:2]
    mh, mw = floor_mask.shape[:2]

    # Resize mask to depth map resolution (nearest-neighbour to keep binary)
    if (mh, mw) != (dh, dw):
        floor_mask_resized = cv2.resize(
            floor_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        floor_mask_resized = floor_mask.astype(bool)

    floor_y, floor_x = np.where(floor_mask_resized)
    floor_d = depth_map[floor_y, floor_x]

    # Filter near-zero depths
    valid = floor_d > 0.01
    floor_x, floor_y, floor_d = floor_x[valid], floor_y[valid], floor_d[valid]

    if len(floor_d) < MIN_FLOOR_PIXELS:
        return None

    # Convert to inverse depth to linearise the plane relationship
    floor_inv_d = 1.0 / floor_d

    # Least-squares fit: A @ [alpha, beta, gamma] = floor_inv_d
    A = np.c_[floor_x, floor_y, np.ones_like(floor_x)]
    coeffs, _, _, _ = np.linalg.lstsq(A, floor_inv_d, rcond=None)
    alpha, beta, gamma = coeffs
    return float(alpha), float(beta), float(gamma)


def expected_floor_depth(
    x_depth: int,
    y_depth: int,
    plane: tuple[float, float, float],
) -> float:
    """
    Query the expected floor depth at a pixel in depth-map space.

    Parameters
    ----------
    x_depth, y_depth : int
        Pixel coordinates in depth-map space.
    plane : (alpha, beta, gamma)
        Fitted plane coefficients.

    Returns
    -------
    expected depth (same units as the original depth map)
    """
    alpha, beta, gamma = plane
    inv_d = alpha * x_depth + beta * y_depth + gamma
    return 1.0 / inv_d if inv_d > 0 else float("inf")
