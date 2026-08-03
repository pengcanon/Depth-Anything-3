"""
feature_extraction.py
---------------------
Extract 2-D feature points from image frames, returning a common `Feature`
dataclass that query_features.py can classify against the floor plane.

Currently implements:
  PoseExtractor  – YOLO-based pose keypoints (17 COCO keypoints per person)

Designed to accept additional extractor types in the future, e.g.:
  BodyMaskExtractor  – contact points derived from a segmentation mask

Public API
----------
Feature            – dataclass: a single 2-D point with name, position, confidence
FeatureExtractor   – abstract base class; subclass and implement extract_batch()
PoseExtractor      – wraps YOLOv8-pose; returns body keypoints as Feature objects

Usage
-----
    extractor = PoseExtractor("yolov8m-pose.pt")
    # returns list[list[list[Feature]]]
    #   dim-0: per image, dim-1: per person, dim-2: per feature point
    batch_features = extractor.extract_batch(image_paths)
    frame_features = batch_features[0]   # first image
    person_features = frame_features[0]  # first person
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ── YOLO/COCO keypoint catalogue ─────────────────────────────────────────────

POSE_KEYPOINT_NAMES: dict[int, str] = {
    0:  "Nose",
    1:  "L-Eye",   2:  "R-Eye",
    3:  "L-Ear",   4:  "R-Ear",
    5:  "L-Shoulder", 6:  "R-Shoulder",
    7:  "L-Elbow", 8:  "R-Elbow",
    9:  "L-Wrist", 10: "R-Wrist",
    11: "L-Hip",   12: "R-Hip",
    13: "L-Knee",  14: "R-Knee",
    15: "L-Ankle", 16: "R-Ankle",
}

# Body joints only (skip face keypoints) — used as the default query set
BODY_JOINT_IDS: list[int] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# ── Common feature dataclass ─────────────────────────────────────────────────

@dataclass
class Feature:
    """
    A single 2-D feature point in original-image pixel space.

    Attributes
    ----------
    name       : Human-readable label (e.g. "L-Ankle", "R-Wrist").
    x, y       : Pixel coordinates in the original (un-scaled) image.
    confidence : Detection confidence in [0, 1].
    source     : Tag identifying which extractor produced this feature
                 (e.g. "pose", "body_mask").
    metadata   : Optional dict for extractor-specific extras
                 (e.g. {"joint_id": 15} for pose keypoints).
    """
    name:       str
    x:          int
    y:          int
    confidence: float
    source:     str = "unknown"
    metadata:   dict = field(default_factory=dict)


# ── Abstract base class ───────────────────────────────────────────────────────

class FeatureExtractor(ABC):
    """
    Base class for all feature extractors.

    Subclasses must implement `extract_batch`, which processes a list of image
    file paths and returns a nested list:
        list[        # per image
          list[      # per detected instance (person, object, …)
            list[Feature]
          ]
        ]
    """

    @abstractmethod
    def extract_batch(
        self,
        image_paths: list[str],
    ) -> list[list[list[Feature]]]:
        """
        Extract features from a batch of images.

        Returns
        -------
        results[i][p] : list[Feature]
            Feature points for person/instance `p` in image `i`.
        """

    def extract_single(self, image_path: str) -> list[list[Feature]]:
        """Convenience wrapper for a single image."""
        return self.extract_batch([image_path])[0]


# ── Pose extractor (YOLO) ─────────────────────────────────────────────────────

class PoseExtractor(FeatureExtractor):
    """
    Extract 17 COCO-format body keypoints per person using YOLOv8-pose.

    Parameters
    ----------
    model_path  : Path or name of the YOLO pose model (e.g. "yolov8m-pose.pt").
    joint_ids   : Which joint IDs to include in the output.
                  Defaults to BODY_JOINT_IDS (all body joints, no face).
    min_conf    : Minimum per-keypoint confidence to include; below this the
                  Feature is still returned but marked with the actual (low)
                  confidence so callers can filter if needed.
    """

    def __init__(
        self,
        model_path: str = "yolov8m-pose.pt",
        joint_ids:  list[int] | None = None,
        min_conf:   float = 0.0,
    ) -> None:
        from ultralytics import YOLO  # deferred import – not always required
        print(f"[PoseExtractor] Loading YOLO pose model: {model_path}")
        self._model    = YOLO(model_path)
        self._joint_ids = joint_ids if joint_ids is not None else list(BODY_JOINT_IDS)
        self._min_conf  = min_conf

    def extract_batch(
        self,
        image_paths: list[str],
    ) -> list[list[list[Feature]]]:
        """
        Run YOLO pose inference on a batch of images.

        Returns
        -------
        results[img_idx][person_idx] : list[Feature]
            One Feature per requested joint for each detected person.
        """
        yolo_results = self._model(image_paths, verbose=False)
        all_features: list[list[list[Feature]]] = []

        for yolo_res in yolo_results:
            image_features: list[list[Feature]] = []

            if yolo_res.keypoints is None or len(yolo_res.keypoints.xy) == 0:
                all_features.append(image_features)
                continue

            kps_xy   = yolo_res.keypoints.xy    # Tensor [P, 17, 2]
            kps_conf = yolo_res.keypoints.conf  # Tensor [P, 17] or None

            for p_idx, kp_xy in enumerate(kps_xy):
                person_features: list[Feature] = []
                kp_xy_np   = kp_xy.cpu().numpy()  # [17, 2]
                kp_conf_np = (
                    kps_conf[p_idx].cpu().numpy()
                    if kps_conf is not None else np.zeros(17)
                )

                for jid in self._joint_ids:
                    x, y = kp_xy_np[jid]
                    conf = float(kp_conf_np[jid])
                    person_features.append(Feature(
                        name=POSE_KEYPOINT_NAMES.get(jid, str(jid)),
                        x=int(x),
                        y=int(y),
                        confidence=conf,
                        source="pose",
                        metadata={"joint_id": jid},
                    ))

                image_features.append(person_features)

            all_features.append(image_features)

        return all_features
