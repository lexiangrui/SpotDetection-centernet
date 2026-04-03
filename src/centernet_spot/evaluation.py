from __future__ import annotations

from typing import Sequence

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def match_detections_greedy(
    detections: Sequence[dict],
    gt_points: np.ndarray,
    match_radius: float,
) -> dict[str, int | list[float]]:
    gt_points = np.asarray(gt_points, dtype=np.float32)
    if gt_points.ndim == 1:
        gt_points = gt_points.reshape(-1, 2)
    if gt_points.size == 0:
        gt_points = gt_points.reshape(0, 2)

    matched_gt = np.zeros((gt_points.shape[0],), dtype=bool)
    distances: list[float] = []
    tp = 0
    fp = 0

    sorted_detections = sorted(detections, key=lambda det: float(det["score"]), reverse=True)
    for det in sorted_detections:
        if gt_points.shape[0] == 0:
            fp += 1
            continue

        remaining_indices = np.flatnonzero(~matched_gt)
        if remaining_indices.size == 0:
            fp += 1
            continue

        pred_xy = np.asarray([float(det["x"]), float(det["y"])], dtype=np.float32)
        deltas = gt_points[remaining_indices] - pred_xy
        candidate_distances = np.hypot(deltas[:, 0], deltas[:, 1])
        best_local_idx = int(candidate_distances.argmin())
        best_gt_idx = int(remaining_indices[best_local_idx])
        best_distance = float(candidate_distances[best_local_idx])

        if best_distance <= match_radius:
            matched_gt[best_gt_idx] = True
            tp += 1
            distances.append(best_distance)
        else:
            fp += 1

    fn = int(gt_points.shape[0] - tp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "distances": distances,
    }


def summarize_detection_metrics(
    tp: int,
    fp: int,
    fn: int,
    distances: Sequence[float],
    score_threshold: float,
) -> dict[str, float | int | None]:
    precision = _safe_divide(float(tp), float(tp + fp))
    recall = _safe_divide(float(tp), float(tp + fn))
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    if distances:
        dist_np = np.asarray(distances, dtype=np.float32)
        mean_loc_error: float | None = float(dist_np.mean())
        median_loc_error: float | None = float(np.median(dist_np))
        p90_loc_error: float | None = float(np.percentile(dist_np, 90))
    else:
        mean_loc_error = None
        median_loc_error = None
        p90_loc_error = None

    return {
        "score_threshold": float(score_threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_loc_error": mean_loc_error,
        "median_loc_error": median_loc_error,
        "p90_loc_error": p90_loc_error,
    }


def evaluate_threshold_sweep(
    predictions_by_image: Sequence[Sequence[dict]],
    gt_points_by_image: Sequence[np.ndarray],
    match_radii: Sequence[float],
    score_thresholds: Sequence[float],
) -> list[dict[str, float | int | None]]:
    if not (len(predictions_by_image) == len(gt_points_by_image) == len(match_radii)):
        raise ValueError("Predictions, ground truth points, and match radii must have the same length.")

    metrics_by_threshold: list[dict[str, float | int | None]] = []
    for threshold in score_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_distances: list[float] = []

        for detections, gt_points, match_radius in zip(predictions_by_image, gt_points_by_image, match_radii):
            filtered_detections = [
                det for det in detections if float(det["score"]) >= float(threshold)
            ]
            matched = match_detections_greedy(filtered_detections, gt_points, float(match_radius))
            total_tp += int(matched["tp"])
            total_fp += int(matched["fp"])
            total_fn += int(matched["fn"])
            all_distances.extend(float(dist) for dist in matched["distances"])

        metrics = summarize_detection_metrics(
            total_tp,
            total_fp,
            total_fn,
            all_distances,
            float(threshold),
        )
        metrics["pred_count"] = int(total_tp + total_fp)
        metrics["gt_count"] = int(total_tp + total_fn)
        metrics_by_threshold.append(metrics)

    return metrics_by_threshold


def compute_average_precision(
    predictions_by_image: Sequence[Sequence[dict]],
    gt_points_by_image: Sequence[np.ndarray],
    match_radii: Sequence[float],
) -> dict[str, float | int | list[float]]:
    if not (len(predictions_by_image) == len(gt_points_by_image) == len(match_radii)):
        raise ValueError("Predictions, ground truth points, and match radii must have the same length.")

    normalized_gt_points: list[np.ndarray] = []
    matched_gt_flags: list[np.ndarray] = []
    flat_predictions: list[tuple[float, int, float, float]] = []

    total_gt = 0
    for image_idx, (detections, gt_points) in enumerate(zip(predictions_by_image, gt_points_by_image)):
        gt_points_np = np.asarray(gt_points, dtype=np.float32)
        if gt_points_np.ndim == 1:
            gt_points_np = gt_points_np.reshape(-1, 2)
        if gt_points_np.size == 0:
            gt_points_np = gt_points_np.reshape(0, 2)

        normalized_gt_points.append(gt_points_np)
        matched_gt_flags.append(np.zeros((gt_points_np.shape[0],), dtype=bool))
        total_gt += int(gt_points_np.shape[0])

        for det in detections:
            flat_predictions.append(
                (
                    float(det["score"]),
                    image_idx,
                    float(det["x"]),
                    float(det["y"]),
                )
            )

    flat_predictions.sort(key=lambda item: item[0], reverse=True)

    if total_gt == 0:
        return {
            "ap": 0.0,
            "total_gt": 0,
            "total_predictions": len(flat_predictions),
            "precision_curve": [],
            "recall_curve": [],
            "score_curve": [],
        }

    tp_flags = np.zeros((len(flat_predictions),), dtype=np.float32)
    fp_flags = np.zeros((len(flat_predictions),), dtype=np.float32)
    score_curve: list[float] = []

    for det_idx, (score, image_idx, pred_x, pred_y) in enumerate(flat_predictions):
        score_curve.append(score)
        gt_points = normalized_gt_points[image_idx]
        if gt_points.shape[0] == 0:
            fp_flags[det_idx] = 1.0
            continue

        remaining_indices = np.flatnonzero(~matched_gt_flags[image_idx])
        if remaining_indices.size == 0:
            fp_flags[det_idx] = 1.0
            continue

        pred_xy = np.asarray([pred_x, pred_y], dtype=np.float32)
        deltas = gt_points[remaining_indices] - pred_xy
        candidate_distances = np.hypot(deltas[:, 0], deltas[:, 1])
        best_local_idx = int(candidate_distances.argmin())
        best_gt_idx = int(remaining_indices[best_local_idx])
        best_distance = float(candidate_distances[best_local_idx])

        if best_distance <= float(match_radii[image_idx]):
            matched_gt_flags[image_idx][best_gt_idx] = True
            tp_flags[det_idx] = 1.0
        else:
            fp_flags[det_idx] = 1.0

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(fp_flags)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    recall = tp_cumsum / float(total_gt)

    ap = 0.0
    prev_recall = 0.0
    for is_tp, prec, rec in zip(tp_flags.tolist(), precision.tolist(), recall.tolist()):
        if is_tp > 0.5:
            ap += (rec - prev_recall) * prec
            prev_recall = rec

    return {
        "ap": float(ap),
        "total_gt": int(total_gt),
        "total_predictions": int(len(flat_predictions)),
        "precision_curve": [float(value) for value in precision.tolist()],
        "recall_curve": [float(value) for value in recall.tolist()],
        "score_curve": [float(value) for value in score_curve],
    }


def select_best_threshold_metrics(
    metrics_by_threshold: Sequence[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    if not metrics_by_threshold:
        raise ValueError("metrics_by_threshold must not be empty.")

    def sort_key(metrics: dict[str, float | int | None]) -> tuple[float, float, float]:
        f1 = float(metrics["f1"])
        mean_loc_error = (
            float(metrics["mean_loc_error"])
            if metrics["mean_loc_error"] is not None
            else float("inf")
        )
        precision = float(metrics["precision"])
        return (f1, -mean_loc_error, precision)

    return dict(max(metrics_by_threshold, key=sort_key))
