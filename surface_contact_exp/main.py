"""
main.py
-------
Entry point for the surface-contact detection pipeline.

Usage
-----
    python main.py --video videos/fall3.avi
    python main.py --video videos/fall3.avi --output out_fall3.avi
    python main.py --video videos/fall3.avi --contact-thresh 0.15 --close-thresh 0.40

The program:
  1. Reads every frame from the input video.
  2. Batches frames (--batch-size) and runs DA3 depth inference.
  3. On the FIRST frame, segments the floor with SAM and fits the depth plane.
     The same plane is reused for all subsequent frames
     (re-fit every --refit-every frames if the scene changes).
  4. Runs YOLO pose estimation to find body keypoints.
  5. Classifies each keypoint against the floor plane.
  6. Writes an annotated output video with colour-coded keypoints:
       red    – CONTACT  (touching the floor)
       yellow – CLOSE    (near the floor)
       green  – SAFE     (clearly above the floor)
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from depth_map          import load_model, get_depth_maps
from fit_floor          import load_sam, segment_floor, fit_floor_plane
from feature_extraction import PoseExtractor, BODY_JOINT_IDS
from floor_normal       import FloorNormalEstimator
from query_features     import classify_features, draw_features

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SAM_CHECKPOINT  = os.path.join(os.path.dirname(__file__), "..", "sam_vit_h_4b8939.pth")
DEFAULT_YOLO_MODEL      = "yolov8m-pose.pt"
DEFAULT_DA3_MODEL       = "depth-anything/DA3NESTED-GIANT-LARGE"
DEFAULT_BATCH           = 1
DEFAULT_REFIT_EVERY     = 0    # default: fit floor once on the first frame
DEFAULT_CONTACT_THRESH  = 0.15
DEFAULT_CLOSE_THRESH    = 0.40
DEFAULT_MIN_CONF        = 0.05
DEFAULT_FLOOR_NORMAL_Z  = None
DEFAULT_MAX_FRAMES      = None
MAX_PROCESS_DIMENSION   = 1500
LARGE_FRAME_SCALE       = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surface-contact detection from video or image.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Input video path")
    src.add_argument("--image", help="Input image path (single-frame mode)")
    p.add_argument("--output",          default=None,   help="Output path (default: <input>_contact.<ext>)")
    p.add_argument("--sam",             default=DEFAULT_SAM_CHECKPOINT, help="SAM checkpoint path")
    p.add_argument("--yolo",            default=DEFAULT_YOLO_MODEL,     help="YOLO pose model")
    p.add_argument("--da3",             default=DEFAULT_DA3_MODEL,      help="DA3 model name")
    p.add_argument("--batch-size",      type=int, default=DEFAULT_BATCH)
    p.add_argument("--max-frames",      type=int, default=DEFAULT_MAX_FRAMES,
                   help="Maximum number of video frames to process (video mode only)")
    p.add_argument("--refit-every",     type=int, default=DEFAULT_REFIT_EVERY,
                   help="Re-fit floor plane every N frames (0 = fit once on the first frame)")
    p.add_argument("--contact-thresh",  type=float, default=DEFAULT_CONTACT_THRESH,
                   help="Absolute depth difference for CONTACT")
    p.add_argument("--close-thresh",    type=float, default=DEFAULT_CLOSE_THRESH,
                   help="Absolute depth difference for CLOSE")
    p.add_argument("--min-conf",        type=float, default=DEFAULT_MIN_CONF,
                   help="Minimum YOLO keypoint confidence")
    p.add_argument("--floor-normal-z",  type=float, default=DEFAULT_FLOOR_NORMAL_Z,
                   help="Optional z component of the floor normal used to scale depth differences")
    p.add_argument("--joints",          nargs="+", type=int, default=BODY_JOINT_IDS,
                   help="YOLO joint IDs to analyse (default: all body joints)")
    return p.parse_args()


def frames_from_video(cap: cv2.VideoCapture) -> list[np.ndarray]:
    """Read all frames as BGR numpy arrays."""
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(_maybe_downscale_frame(frame))
    return frames


def _maybe_downscale_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Halve large frames to reduce SAM/DSINE/YOLO startup and per-frame cost."""
    h, w = frame_bgr.shape[:2]
    if max(h, w) <= MAX_PROCESS_DIMENSION:
        return frame_bgr
    return cv2.resize(
        frame_bgr,
        (int(w * LARGE_FRAME_SCALE), int(h * LARGE_FRAME_SCALE)),
        interpolation=cv2.INTER_AREA,
    )


def save_frames_as_tmp_images(frames_bgr: list[np.ndarray], tmp_dir: str) -> list[str]:
    """Write a batch of BGR frames to disk so DA3 can load them."""
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for i, f in enumerate(frames_bgr):
        p = os.path.join(tmp_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(p, f)
        paths.append(p)
    return paths


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    # ── load models (shared by both modes) ────────────────────────────────────
    print("[main] Loading models ...")
    da3_model      = load_model(args.da3, device=device)
    sam_pred       = load_sam(args.sam, device=device)
    normal_estimator = FloorNormalEstimator(device=device)
    pose_extractor = PoseExtractor(args.yolo, joint_ids=args.joints, min_conf=0.0)
    print("[main] All models loaded.")

    if args.image:
        process_image(args, da3_model, sam_pred, normal_estimator, pose_extractor)
    else:
        process_video(args, da3_model, sam_pred, normal_estimator, pose_extractor, device)


def _save_floor_model_artifacts(base_path: str, floor_mask: np.ndarray, floor_normal: np.ndarray) -> None:
    """Save the first-frame floor mask and estimated floor-normal vector."""
    mask_path = base_path + "_floor_mask.png"
    normal_path = base_path + "_floor_normal.txt"
    np.save(base_path + "_floor_normal.npy", floor_normal.astype(np.float32))
    cv2.imwrite(mask_path, floor_mask.astype(np.uint8) * 255)
    with open(normal_path, "w", encoding="utf-8") as f:
        f.write("# floor normal in camera coordinates (unit vector)\n")
        f.write(f"{floor_normal[0]:.8f} {floor_normal[1]:.8f} {floor_normal[2]:.8f}\n")


def process_image(
    args,
    da3_model,
    sam_pred,
    normal_estimator: FloorNormalEstimator,
    pose_extractor: PoseExtractor,
) -> None:
    """Single-image mode: annotate and save one image."""
    img_path = args.image
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Output path
    if args.output is None:
        base, ext = os.path.splitext(img_path)
        out_path = base + "_contact" + (ext or ".jpg")
    else:
        out_path = args.output
    out_base, _ = os.path.splitext(out_path)

    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    frame_bgr = _maybe_downscale_frame(frame_bgr)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_bgr.shape[:2]

    # Depth map
    tmp_dir = os.path.join(os.path.dirname(img_path), "_tmp_single")
    tmp_path = save_frames_as_tmp_images([frame_bgr], tmp_dir)[0]
    prediction = get_depth_maps([tmp_path], model=da3_model)
    depth_map  = prediction.depth[0]

    # Floor segmentation + plane fit
    floor_mask = segment_floor(frame_rgb, sam_pred)
    plane = fit_floor_plane(floor_mask, depth_map) if floor_mask is not None else None
    normal_map = normal_estimator.estimate_normal_map(frame_bgr, input_format="bgr")
    floor_normal_z = args.floor_normal_z
    if floor_mask is not None and floor_normal_z is None:
        floor_normal = normal_estimator.estimate_floor_normal(
            normal_map=normal_map,
            floor_mask=floor_mask,
        )
        floor_normal_z = float(floor_normal[2])
    if plane:
        print(f"[image] Floor plane: α={plane[0]:.6f}  β={plane[1]:.6f}  γ={plane[2]:.6f}")
    else:
        print("[image] Floor plane fit failed; contact detection disabled.")
    if floor_normal_z is not None:
        print(f"[image] Floor normal z: {floor_normal_z:.6f}")

    # Pose + classify
    frame_persons = pose_extractor.extract_batch([tmp_path])[0]
    annotated = frame_bgr.copy()
    all_results: list = []
    for person_features in frame_persons:
        results = classify_features(
            features=person_features,
            depth_map=depth_map,
            plane=plane,
            orig_shape=(height, width),
            min_conf=args.min_conf,
            contact_thresh=args.contact_thresh,
            close_thresh=args.close_thresh,
            floor_normal_z=floor_normal_z,
        )
        all_results.extend(results)
        annotated = draw_features(annotated, results)

    _draw_legend(annotated)
    cv2.imwrite(out_path, annotated)

    # Save depth artifacts for inspection
    depth_npy_path = out_base + "_depth.npy"
    depth_vis_path = out_base + "_depth_vis.png"
    np.save(depth_npy_path, depth_map.astype(np.float32))
    finite_mask = np.isfinite(depth_map)
    if finite_mask.any():
        d_min = float(depth_map[finite_mask].min())
        d_max = float(depth_map[finite_mask].max())
        if d_max > d_min:
            depth_norm = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.float32)
    else:
        depth_norm = np.zeros_like(depth_map, dtype=np.float32)
    depth_vis_u8 = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(depth_vis_path, depth_vis_color)

    # Save floor mask artifact when available
    floor_mask_path = out_base + "_floor_mask.png"
    if floor_mask is not None:
        cv2.imwrite(floor_mask_path, floor_mask.astype(np.uint8) * 255)

    # Save surface-normal artifacts
    normal_npy_path = out_base + "_normal.npy"
    normal_vis_path = out_base + "_normal_vis.png"
    np.save(normal_npy_path, normal_map.astype(np.float32))
    normal_vis_rgb = ((np.clip(normal_map, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)
    normal_vis_bgr = cv2.cvtColor(normal_vis_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(normal_vis_path, normal_vis_bgr)

    import shutil
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    print(f"[image] Saved to: {out_path}")
    print(f"[image] Depth map saved: {depth_npy_path}")
    print(f"[image] Depth visualization saved: {depth_vis_path}")
    print(f"[image] Surface normal map saved: {normal_npy_path}")
    print(f"[image] Surface normal visualization saved: {normal_vis_path}")
    if floor_mask is not None:
        print(f"[image] Floor mask saved: {floor_mask_path}")

    # ── per-joint depth report ─────────────────────────────────────────────────
    from query_features import ContactStatus
    print(f"\n{'Joint':<20} {'Actual d':>10} {'Floor d':>10} {'|Δd|':>10}  Status")
    print("-" * 60)
    for r in all_results:
        if r.status == ContactStatus.UNKNOWN:
            continue
        print(f"{r.feature.name:<20} {r.actual_depth:>10.4f} {r.expected_depth:>10.4f} "
              f"{abs(r.depth_diff):>10.4f}  {r.status.name}")


def process_video(
    args,
    da3_model,
    sam_pred,
    normal_estimator: FloorNormalEstimator,
    pose_extractor: PoseExtractor,
    device,
) -> None:
    """Video mode: annotate every frame and write output video."""
    # ── output path ────────────────────────────────────────────────────────────
    if args.output is None:
        base, _ = os.path.splitext(args.video)
        args.output = base + "_contact.avi"
    out_base, _ = os.path.splitext(args.output)

    # ── read video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[main] Video: {raw_width}x{raw_height}  {fps:.1f} fps  {total} frames")

    all_frames = frames_from_video(cap)
    cap.release()

    if not all_frames:
        raise ValueError(f"Video has no readable frames: {args.video}")

    if args.max_frames is not None:
        all_frames = all_frames[: max(0, args.max_frames)]
        print(f"[main] Limiting processing to first {len(all_frames)} frames")

    height, width = all_frames[0].shape[:2]
    if (height, width) != (raw_height, raw_width):
        print(f"[main] Downscaled processing resolution to: {width}x{height}")

    # ── video writer ───────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # ── temporary directory for DA3 image loading ──────────────────────────────
    tmp_dir = os.path.join(os.path.dirname(args.video), "_tmp_frames")

    # ── main processing loop ───────────────────────────────────────────────────
    plane: tuple[float, float, float] | None = None
    floor_normal_z = args.floor_normal_z
    floor_normal_vec: np.ndarray | None = None
    floor_artifacts_saved = False
    n = len(all_frames)

    for batch_start in range(0, n, args.batch_size):
        batch_bgr   = all_frames[batch_start : batch_start + args.batch_size]
        batch_rgb   = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
        batch_paths = save_frames_as_tmp_images(batch_bgr, tmp_dir)

        print(f"[main] Processing frames {batch_start}–{batch_start+len(batch_bgr)-1} ...")

        # ── DA3 depth inference + pose extraction (timed together) ────────────
        t_batch_start = time.perf_counter()
        prediction = get_depth_maps(batch_paths, model=da3_model, device=device)
        batch_features = pose_extractor.extract_batch(batch_paths)
        batch_size_actual = len(batch_bgr)
        t_inference_per_frame = (time.perf_counter() - t_batch_start) / batch_size_actual

        # ── per-frame processing ───────────────────────────────────────────────
        for local_idx, (frame_bgr, frame_rgb, depth_map, frame_person_features) in enumerate(
            zip(batch_bgr, batch_rgb, prediction.depth, batch_features)
        ):
            global_idx = batch_start + local_idx

            # Re-fit floor plane if needed
            refit = plane is None or (args.refit_every > 0 and global_idx % args.refit_every == 0)
            if refit:
                floor_mask = segment_floor(frame_rgb, sam_pred)
                if floor_mask is not None:
                    plane = fit_floor_plane(floor_mask, depth_map)
                    floor_normal_vec = normal_estimator.estimate_floor_normal(
                        normal_map=normal_estimator.estimate_normal_map(frame_bgr, input_format="bgr"),
                        floor_mask=floor_mask,
                    )
                    if args.floor_normal_z is None:
                        floor_normal_z = float(floor_normal_vec[2])
                    if not floor_artifacts_saved:
                        _save_floor_model_artifacts(out_base, floor_mask, floor_normal_vec)
                        floor_artifacts_saved = True
                    if plane is not None:
                        print(f"  [frame {global_idx}] Floor plane fitted: "
                              f"α={plane[0]:.6f}  β={plane[1]:.6f}  γ={plane[2]:.6f}")
                        if floor_normal_vec is not None:
                            print(
                                f"  [frame {global_idx}] Floor normal: "
                                f"[{floor_normal_vec[0]:.6f}, {floor_normal_vec[1]:.6f}, {floor_normal_vec[2]:.6f}]"
                            )
                        elif floor_normal_z is not None:
                            print(f"  [frame {global_idx}] Floor normal z: {floor_normal_z:.6f}")
                    else:
                        print(f"  [frame {global_idx}] Floor plane fit failed.")
                else:
                    print(f"  [frame {global_idx}] SAM floor segmentation failed.")

            # Classify features and draw
            t_classify_start = time.perf_counter()
            annotated = frame_bgr.copy()
            for person_features in frame_person_features:
                results = classify_features(
                    features=person_features,
                    depth_map=depth_map,
                    plane=plane,
                    orig_shape=(height, width),
                    min_conf=args.min_conf,
                    contact_thresh=args.contact_thresh,
                    close_thresh=args.close_thresh,
                    floor_normal_z=floor_normal_z,
                )
                annotated = draw_features(annotated, results)

            # FPS = 1 / (depth+pose time per frame + classification time)
            t_total = t_inference_per_frame + (time.perf_counter() - t_classify_start)
            proc_fps = 1.0 / t_total if t_total > 0 else 0.0
            _draw_legend(annotated)
            _draw_fps(annotated, proc_fps, global_idx, total)
            writer.write(annotated)

    writer.release()

    # Clean up temp images
    import shutil
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    print(f"[main] Done. Output saved to: {args.output}")


def _draw_fps(frame: np.ndarray, fps: float, frame_idx: int, total: int) -> None:
    """Draw processing FPS and frame counter in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:5.1f}   frame {frame_idx + 1}/{total}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = w - tw - 10
    y = 22
    # subtle dark background for readability
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray) -> None:
    """Draw a small colour legend in the top-left corner."""
    items = [
        ((0,   0,   255), "CONTACT"),
        ((0,   200, 255), "CLOSE"),
        ((0,   200,  50), "SAFE"),
    ]
    x0, y0, dy = 10, 20, 22
    for i, (color, label) in enumerate(items):
        y = y0 + i * dy
        cv2.circle(frame, (x0 + 8, y), 7, color, -1)
        cv2.circle(frame, (x0 + 8, y), 7, (0, 0, 0), 1)
        cv2.putText(frame, label, (x0 + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
