"""
query_features.py
-----------------
Classify 2-D feature points (produced by feature_extraction.py) against the
fitted floor depth plane and render colour-coded overlays.

  ContactStatus.CONTACT  – touching floor  → red
  ContactStatus.CLOSE    – very close      → yellow
  ContactStatus.SAFE     – safely above    → green

Public API
----------
classify_features(features, depth_map, plane, orig_shape, ...)
    -> list[FeatureResult]

draw_features(frame_bgr, results, radius=8)
    -> annotated BGR frame
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from feature_extraction import Feature


class ContactStatus(Enum):
    CONTACT = auto()   # red
    CLOSE   = auto()   # yellow
    SAFE    = auto()   # green
    UNKNOWN = auto()   # grey  (low confidence / no floor plane)


# BGR colours per status
STATUS_COLOR = {
    ContactStatus.CONTACT: (0,   0,   255),  # red
    ContactStatus.CLOSE:   (0,   200, 255),  # yellow
    ContactStatus.SAFE:    (0,   200,  50),  # green
    ContactStatus.UNKNOWN: (128, 128, 128),  # grey
}


@dataclass
class FeatureResult:
    """Classification result for a single feature point."""
    feature:        Feature        # original feature point
    actual_depth:   float
    expected_depth: float
    depth_diff:     float          # expected - actual  (+ = above floor)
    status:         ContactStatus


# Backward-compatibility alias
KeypointResult = FeatureResult


def classify_features(
    features:       list[Feature],
    depth_map:      np.ndarray,          # [Hd, Wd] float32  (depth-map space)
    plane:          tuple[float, float, float] | None,
    orig_shape:     tuple[int, int],     # (H_orig, W_orig)
    min_conf:       float = 0.05,
    contact_thresh: float = 0.15,        # absolute depth units
    close_thresh:   float = 0.40,        # absolute depth units
    floor_normal_z: float | None = None,
) -> list[FeatureResult]:
    """
    Classify a list of Feature points against the fitted floor plane.

    Parameters
    ----------
    features       : Feature objects in original-image pixel space.
    depth_map      : DA3 depth map (model output resolution).
    plane          : (alpha, beta, gamma) from fit_floor.fit_floor_plane().
    orig_shape     : (H, W) of the original image.
    contact_thresh : |depth_diff| <= this  →  CONTACT.
    close_thresh   : depth_diff <= this    →  CLOSE  (otherwise SAFE).
    floor_normal_z : Optional floor-normal z component. When provided, the
                     depth difference is scaled by |floor_normal_z| so the
                     threshold tracks floor tilt relative to the optic axis.
    """
    results: list[FeatureResult] = []
    diff_scale = abs(float(floor_normal_z)) if floor_normal_z is not None else 1.0

    orig_h, orig_w = orig_shape
    dh, dw = depth_map.shape[:2]
    sx = dw / orig_w
    sy = dh / orig_h

    for feat in features:
        if feat.confidence < min_conf or plane is None:
            results.append(FeatureResult(
                feature=feat,
                actual_depth=float("nan"), expected_depth=float("nan"),
                depth_diff=float("nan"), status=ContactStatus.UNKNOWN,
            ))
            continue

        # Scale original-image pixel → depth-map pixel
        x_d = int(np.clip(feat.x * sx, 0, dw - 1))
        y_d = int(np.clip(feat.y * sy, 0, dh - 1))

        alpha, beta, gamma = plane
        inv_d    = alpha * x_d + beta * y_d + gamma
        expected = 1.0 / inv_d if inv_d > 0 else float("inf")
        actual   = float(depth_map[y_d, x_d])
        diff     = (expected - actual) * diff_scale

        if abs(diff) <= contact_thresh:
            status = ContactStatus.CONTACT
        elif diff <= close_thresh:
            status = ContactStatus.CLOSE
        else:
            status = ContactStatus.SAFE

        results.append(FeatureResult(
            feature=feat,
            actual_depth=actual, expected_depth=expected,
            depth_diff=diff, status=status,
        ))

    return results


def draw_features(
    frame_bgr: np.ndarray,
    results:   list[FeatureResult],
    radius:    int = 8,
    thickness: int = -1,
    label:     bool = True,
) -> np.ndarray:
    """
    Overlay classified feature points on a BGR frame.

    Returns a copy of the frame with colour-coded circles.
    """
    out = frame_bgr.copy()
    for r in results:
        if r.status == ContactStatus.UNKNOWN:
            continue
        color = STATUS_COLOR[r.status]
        cx, cy = r.feature.x, r.feature.y
        cv2.circle(out, (cx, cy), radius, color, thickness)
        cv2.circle(out, (cx, cy), radius, (0, 0, 0), 1)  # thin black border
        if label:
            cv2.putText(
                out, r.feature.name,
                (cx + radius + 2, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
    return out


# Backward-compatibility alias
draw_keypoints = draw_features
