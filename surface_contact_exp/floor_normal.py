"""
floor_normal.py
---------------
Estimate surface normals and derive a floor normal vector in camera coordinates.

This module is intentionally backend-agnostic. Multiple normal-estimation
backends can be added later; for now, DSINE (via torch hub) is implemented.

Primary class
-------------
FloorNormalEstimator
    - estimate_normal_map(image, input_format="bgr") -> (H, W, 3) float32
    - estimate_floor_normal(normal_map, floor_mask=None, sample_point=None)
    - estimate_from_image(image, floor_mask=None, input_format="bgr")

Notes
-----
- DSINE output is interpreted as per-pixel normal vectors in camera space.
- The returned floor normal is a unit vector.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch

from fit_floor import load_sam, segment_floor


NormalBackend = Literal["dsine"]
DEFAULT_SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "sam_vit_h_4b8939.pth")


@dataclass
class FloorNormalResult:
    """Container for floor-normal estimation outputs."""

    floor_normal: np.ndarray
    normal_map: np.ndarray


class FloorNormalEstimator:
    """
    Estimate floor normal vectors from images.

    Parameters
    ----------
    backend : {"dsine"}
        Normal-estimation backend. Only DSINE is implemented for now.
    device : torch.device | str | None
        Device for model execution. Auto-detected when None.
    trust_repo : bool
        Forwarded to torch.hub.load for DSINE hub loading.
    """

    def __init__(
        self,
        backend: NormalBackend = "dsine",
        device: torch.device | str | None = None,
        trust_repo: bool = True,
    ) -> None:
        self.backend = backend
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.trust_repo = trust_repo
        self.model = self._load_backend_model(backend)

    def _load_backend_model(self, backend: NormalBackend):
        if backend == "dsine":
            return self._load_dsine_model()
        raise ValueError(f"Unsupported normal backend: {backend}")

    def _load_dsine_model(self):
        """Load DSINE from torch hub."""
        try:
            # The DSINE hub wrapper exposes an `infer_cv2` helper used below.
            model = torch.hub.load(
                "hugoycj/DSINE-hub",
                "DSINE",
                trust_repo=self.trust_repo,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DSINE from torch hub. Ensure network access and "
                "that required packages are installed in the active environment."
            ) from exc

        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    @staticmethod
    def _to_bgr(image: np.ndarray, input_format: Literal["bgr", "rgb"]) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected image with shape (H, W, 3)")
        if input_format == "bgr":
            return image
        if input_format == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise ValueError("input_format must be 'bgr' or 'rgb'")

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(norms, eps)

    def estimate_normal_map(
        self,
        image: np.ndarray,
        input_format: Literal["bgr", "rgb"] = "bgr",
    ) -> np.ndarray:
        """
        Estimate per-pixel normal vectors.

        Returns
        -------
        normal_map : np.ndarray
            Float32 array with shape (H, W, 3), approximately unit-length.
        """
        image_bgr = self._to_bgr(image, input_format=input_format)

        with torch.inference_mode():
            raw = self.model.infer_cv2(image_bgr)[0]

        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()

        # Accept either [3, H, W] or [1, 3, H, W] from DSINE wrappers.
        if raw.ndim == 4 and raw.shape[0] == 1:
            raw = raw[0]

        if raw.ndim != 3 or raw.shape[0] != 3:
            raise ValueError(
                f"Unexpected DSINE output shape: {raw.shape}. Expected [3, H, W]."
            )

        normal_map = raw.transpose(1, 2, 0).astype(np.float32)
        normal_map = self._normalize_vectors(normal_map)
        return normal_map

    def estimate_floor_normal(
        self,
        normal_map: np.ndarray,
        floor_mask: np.ndarray | None = None,
        sample_point: tuple[int, int] | None = None,
        aggregate: Literal["median", "mean"] = "median",
    ) -> np.ndarray:
        """
        Estimate a single floor-normal vector from a normal map.

        Parameters
        ----------
        normal_map : np.ndarray
            Surface-normal map with shape (H, W, 3).
        floor_mask : np.ndarray | None
            Optional floor mask with shape (H, W). If omitted, a fallback region
            at the lower-middle image area is used.
        sample_point : (y, x) | None
            Optional direct sampling point. If provided, this takes precedence.
        aggregate : {"median", "mean"}
            Aggregation method when multiple pixels are used.

        Returns
        -------
        floor_normal : np.ndarray
            Unit vector of shape (3,) in camera coordinates.
        """
        if normal_map.ndim != 3 or normal_map.shape[2] != 3:
            raise ValueError("normal_map must have shape (H, W, 3)")

        h, w, _ = normal_map.shape

        if sample_point is not None:
            y, x = sample_point
            y = int(np.clip(y, 0, h - 1))
            x = int(np.clip(x, 0, w - 1))
            vec = normal_map[y, x]
            return self._normalize_vectors(vec.reshape(1, 1, 3))[0, 0]

        if floor_mask is not None:
            if floor_mask.shape[:2] != (h, w):
                floor_mask = cv2.resize(
                    floor_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                floor_mask = floor_mask.astype(bool)
            samples = normal_map[floor_mask]
        else:
            y = int(np.clip(15 * h / 16, 0, h - 1))
            x = int(np.clip(w / 2, 0, w - 1))
            vec = normal_map[y, x]
            return self._normalize_vectors(vec.reshape(1, 1, 3))[0, 0]

        if samples.size == 0:
            raise ValueError("No valid normal samples found for floor estimation.")

        finite_mask = np.isfinite(samples).all(axis=1)
        samples = samples[finite_mask]
        if len(samples) == 0:
            raise ValueError("All sampled normals are non-finite.")

        if aggregate == "median":
            vec = np.median(samples, axis=0)
        elif aggregate == "mean":
            vec = np.mean(samples, axis=0)
        else:
            raise ValueError("aggregate must be 'median' or 'mean'")

        vec = self._normalize_vectors(vec.reshape(1, 1, 3))[0, 0]
        return vec.astype(np.float32)

    def estimate_from_image(
        self,
        image: np.ndarray,
        floor_mask: np.ndarray | None = None,
        sample_point: tuple[int, int] | None = None,
        input_format: Literal["bgr", "rgb"] = "bgr",
        aggregate: Literal["median", "mean"] = "median",
    ) -> FloorNormalResult:
        """Convenience wrapper: image -> normal map + floor normal."""
        normal_map = self.estimate_normal_map(image=image, input_format=input_format)
        floor_normal = self.estimate_floor_normal(
            normal_map=normal_map,
            floor_mask=floor_mask,
            sample_point=sample_point,
            aggregate=aggregate,
        )
        return FloorNormalResult(floor_normal=floor_normal, normal_map=normal_map)


def _normal_map_to_bgr_vis(normal_map: np.ndarray) -> np.ndarray:
    """Map normals in [-1, 1] to an 8-bit RGB visualization, then convert to BGR."""
    vis_rgb = ((normal_map + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate floor normal using DSINE.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--sam",
        default=DEFAULT_SAM_CHECKPOINT,
        help="SAM checkpoint path for floor-mask extraction",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("surface_contact_exp", "output"),
        help="Directory to write output artifacts",
    )
    parser.add_argument(
        "--aggregate",
        choices=["median", "mean"],
        default="median",
        help="Aggregation method for floor-normal extraction",
    )
    parser.add_argument(
        "--sample-point",
        nargs=2,
        type=int,
        metavar=("Y", "X"),
        default=None,
        help="Optional pixel location for direct floor-normal sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    estimator = FloorNormalEstimator(backend="dsine")
    sample_point = tuple(args.sample_point) if args.sample_point is not None else None
    floor_mask = None

    if sample_point is None:
        sam_predictor = load_sam(args.sam, device=estimator.device)
        floor_mask = segment_floor(image_rgb, sam_predictor)
        if floor_mask is None:
            print("[floor_normal] SAM floor segmentation failed; falling back to point sampling.")

    result = estimator.estimate_from_image(
        image=image_bgr,
        floor_mask=floor_mask,
        sample_point=sample_point,
        input_format="bgr",
        aggregate=args.aggregate,
    )

    normal_vis_path = os.path.join(args.out_dir, f"{stem}_normal_vis.png")
    floor_txt_path = os.path.join(args.out_dir, f"{stem}_floor_normal.txt")
    normal_npy_path = os.path.join(args.out_dir, f"{stem}_normal_map.npy")
    floor_mask_path = os.path.join(args.out_dir, f"{stem}_floor_mask.png")

    cv2.imwrite(normal_vis_path, _normal_map_to_bgr_vis(result.normal_map))
    np.save(normal_npy_path, result.normal_map)
    if floor_mask is not None:
        cv2.imwrite(floor_mask_path, floor_mask.astype(np.uint8) * 255)

    fx, fy, fz = result.floor_normal.tolist()
    with open(floor_txt_path, "w", encoding="utf-8") as f:
        f.write("# floor normal in camera coordinates (unit vector)\n")
        if floor_mask is not None:
            f.write("# source: SAM floor mask aggregation\n")
        elif sample_point is not None:
            f.write(f"# source: sample_point=({sample_point[0]}, {sample_point[1]})\n")
        else:
            h, w = image_bgr.shape[:2]
            f.write(f"# source: fallback sample_point=({int(15 * h / 16)}, {int(w / 2)})\n")
        f.write(f"{fx:.8f} {fy:.8f} {fz:.8f}\n")

    print(f"[floor_normal] image: {args.image}")
    print(f"[floor_normal] floor normal: [{fx:.6f}, {fy:.6f}, {fz:.6f}]")
    if floor_mask is not None:
        print(f"[floor_normal] floor mask: {floor_mask_path}")
    print(f"[floor_normal] normal visualization: {normal_vis_path}")
    print(f"[floor_normal] normal map (.npy): {normal_npy_path}")
    print(f"[floor_normal] floor normal (.txt): {floor_txt_path}")


if __name__ == "__main__":
    main()
