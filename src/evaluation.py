"""
evaluation.py
-------------
Lightweight evaluation scaffold for comparing tracker output against
ground-truth annotations in MOT-Challenge format (frame, id, x, y, w, h, ...).

Honest scope note
------------------
Full MOTA/MOTP computation requires a carefully-specified matching
protocol (per-frame Hungarian matching against ground truth at an IoU
threshold, correct handling of ID-switch bookkeeping across the whole
sequence, distractor/ignore-region handling, etc.). Implementing an
incorrect version of these metrics would be worse than not having them --
it would give false confidence in numbers that don't match the standard
definition used by the MOT benchmark and py-motmetrics.

Rather than ship a subtly-wrong MOTA/MOTP, this module implements:
  * IoU-based frame-by-frame matching between predicted and ground-truth
    boxes (the actual hard part, and the part reused by real MOTA/MOTP).
  * A small set of *correctly defined* summary counts derived from that
    matching: precision, recall, detection F1, and a simple ID-switch
    count (how many times a ground-truth identity's matched predicted ID
    changes between consecutive frames it appears in).

For rigorous MOTA/MOTP/IDF1 reporting, export this module's per-frame
matches into `motmetrics` (https://github.com/cheind/py-motmetrics), which
is the standard, well-tested reference implementation -- see
`to_motmetrics_accumulator()` below for a ready-made bridge if that
package is installed.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import iou

logger = logging.getLogger("surveillance")

BBox = Tuple[float, float, float, float]


@dataclass
class GTBox:
    frame: int
    track_id: int
    bbox: BBox


@dataclass
class PredBox:
    frame: int
    track_id: int
    bbox: BBox
    confidence: float = 1.0


def load_mot_ground_truth(
    path: str,
    mot17_pedestrians_only: bool = False,
) -> List[GTBox]:
    """Load ground truth in MOT-Challenge gt.txt format.

    Format:
        frame, id, x, y, w, h, conf, class, visibility

    When ``mot17_pedestrians_only`` is True, keep only annotations that
    represent valid MOT17 pedestrian targets:

        confidence == 1
        class == 1

    The default remains generic MOT-format loading, so existing callers
    continue to work unchanged.
    """
    boxes: List[GTBox] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row_num, row in enumerate(reader, start=1):
                if len(row) < 6:
                    logger.warning(
                        "Skipping malformed ground-truth row %d in %s "
                        "(need >= 6 columns)",
                        row_num,
                        path,
                    )
                    continue

                try:
                    frame = int(float(row[0]))
                    track_id = int(float(row[1]))
                    x, y, w, h = (float(v) for v in row[2:6])
                except ValueError:
                    logger.warning(
                        "Skipping non-numeric ground-truth row %d in %s",
                        row_num,
                        path,
                    )
                    continue

                # MOT17 gt.txt has:
                # conf = column 7
                # class = column 8
                #
                # For pedestrian evaluation we keep only:
                #   confidence == 1
                #   class == 1
                if mot17_pedestrians_only:
                    if len(row) < 8:
                        logger.warning(
                            "Skipping MOT17 row %d in %s "
                            "(need >= 8 columns)",
                            row_num,
                            path,
                        )
                        continue

                    try:
                        confidence = float(row[6])
                        object_class = int(float(row[7]))
                    except ValueError:
                        logger.warning(
                            "Skipping invalid MOT17 metadata row %d in %s",
                            row_num,
                            path,
                        )
                        continue

                    if confidence != 1.0 or object_class != 1:
                        continue

                boxes.append(
                    GTBox(
                        frame=frame,
                        track_id=track_id,
                        bbox=(x, y, x + w, y + h),
                    )
                )

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Ground-truth file not found: '{path}'"
        ) from exc

    logger.info(
        "Loaded %d ground-truth boxes from %s%s",
        len(boxes),
        path,
        " (MOT17 pedestrians only)"
        if mot17_pedestrians_only
        else "",
    )

    return boxes

@dataclass
class EvaluationResult:
    frames_evaluated: int
    total_gt_boxes: int
    total_pred_boxes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        return (
            f"Frames evaluated: {self.frames_evaluated}\n"
            f"GT boxes: {self.total_gt_boxes} | Predicted boxes: {self.total_pred_boxes}\n"
            f"TP: {self.true_positives} | FP: {self.false_positives} | FN: {self.false_negatives}\n"
            f"Precision: {self.precision:.3f} | Recall: {self.recall:.3f} | F1: {self.f1:.3f}\n"
            f"ID switches: {self.id_switches}\n"
            "(Note: MOTA/MOTP/IDF1 are NOT computed here -- see module "
            "docstring. Precision/Recall/F1/ID-switch-count use standard, "
            "unambiguous definitions and are safe to report as-is.)"
        )


def evaluate(
    ground_truth: List[GTBox],
    predictions: List[PredBox],
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Frame-by-frame greedy IoU matching between predictions and ground truth."""
    gt_by_frame: Dict[int, List[GTBox]] = defaultdict(list)
    pred_by_frame: Dict[int, List[PredBox]] = defaultdict(list)
    for g in ground_truth:
        gt_by_frame[g.frame].append(g)
    for p in predictions:
        pred_by_frame[p.frame].append(p)

    frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    tp = fp = fn = 0
    last_matched_pred_id: Dict[int, int] = {}  # gt_track_id -> last matched pred track_id
    id_switches = 0

    for frame in frames:
        gts = gt_by_frame.get(frame, [])
        preds = pred_by_frame.get(frame, [])
        unmatched_preds = list(range(len(preds)))
        matched_gt_indices = set()

        # Greedy matching by descending IoU (simple, deterministic, and
        # sufficient for the precision/recall/F1/ID-switch counts we report;
        # see module docstring re: full MOTA's optimal-assignment nuance).
        pairs = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                score = iou(g.bbox, p.bbox)
                if score >= iou_threshold:
                    pairs.append((score, gi, pi))
        pairs.sort(key=lambda t: t[0], reverse=True)

        used_gt, used_pred = set(), set()
        for score, gi, pi in pairs:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            matched_gt_indices.add(gi)
            if pi in unmatched_preds:
                unmatched_preds.remove(pi)

            gt_id = gts[gi].track_id
            pred_id = preds[pi].track_id
            if gt_id in last_matched_pred_id and last_matched_pred_id[gt_id] != pred_id:
                id_switches += 1
            last_matched_pred_id[gt_id] = pred_id
            tp += 1

        fn += len(gts) - len(matched_gt_indices)
        fp += len(unmatched_preds)

    return EvaluationResult(
        frames_evaluated=len(frames),
        total_gt_boxes=len(ground_truth),
        total_pred_boxes=len(predictions),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        id_switches=id_switches,
    )


def to_motmetrics_accumulator(ground_truth: List[GTBox], predictions: List[PredBox]):
    """Build a `motmetrics.MOTAccumulator` for rigorous MOTA/MOTP/IDF1.

    Returns None (with a logged warning) if the optional `motmetrics`
    package is not installed, rather than failing the whole evaluation --
    this bridge is a convenience, not a hard dependency of the project.
    """
    try:
        import motmetrics as mm
    except ImportError:
        logger.warning(
            "motmetrics is not installed; skipping full MOTA/MOTP accumulator. "
            "Install with `pip install motmetrics` for standard MOT metrics."
        )
        return None

    import numpy as np

    acc = mm.MOTAccumulator(auto_id=True)
    gt_by_frame: Dict[int, List[GTBox]] = defaultdict(list)
    pred_by_frame: Dict[int, List[PredBox]] = defaultdict(list)
    for g in ground_truth:
        gt_by_frame[g.frame].append(g)
    for p in predictions:
        pred_by_frame[p.frame].append(p)

    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gts = gt_by_frame.get(frame, [])
        preds = pred_by_frame.get(frame, [])
        gt_ids = [g.track_id for g in gts]
        pred_ids = [p.track_id for p in preds]
        distances = mm.distances.iou_matrix(
            [g.bbox for g in gts], [p.bbox for p in preds], max_iou=0.5
        ) if gts and preds else np.empty((len(gts), len(preds)))
        acc.update(gt_ids, pred_ids, distances)

    return acc
