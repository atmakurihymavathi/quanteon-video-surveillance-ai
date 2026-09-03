"""
detector.py
-----------
Person detection using a YOLO-family object detector executed through
OpenCV's DNN module (cv2.dnn).

Why OpenCV DNN + Darknet weights instead of PyTorch/ultralytics?
See README.md "Model Selection" section for the full justification.
In short: it keeps the dependency footprint small (no PyTorch/CUDA
toolchain required), starts fast, runs reliably on CPU-only reviewer
machines, and still gives a modern, purpose-built one-stage detector
(YOLOv4-tiny) with a bundled COCO 'person' class.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .utils import BBox

logger = logging.getLogger("surveillance")

COCO_PERSON_CLASS_ID = 0  # index of 'person' in coco.names


@dataclass
class Detection:
    """A single person detection in one frame."""

    bbox: BBox  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int = COCO_PERSON_CLASS_ID


class DetectorError(RuntimeError):
    """Raised when the detector cannot be initialized or run."""


class PersonDetector:
    """Wraps a Darknet YOLO model loaded via cv2.dnn for person detection.

    Parameters
    ----------
    model_cfg, model_weights, class_names:
        Paths to the Darknet .cfg / .weights files and the COCO class-name
        list. Defaults point at the bundled YOLOv4-tiny model.
    confidence_threshold:
        Minimum class confidence to keep a raw detection before NMS.
    nms_threshold:
        IoU threshold used for non-max suppression.
    input_size:
        Network input resolution (square). 416 is the standard YOLOv4-tiny
        input size and offers a good speed/accuracy trade-off on CPU.
    device:
        'auto' | 'cpu' | 'cuda'. 'auto' will try to use an OpenCV CUDA
        backend if this OpenCV build supports it and a CUDA device is
        present, otherwise it transparently falls back to CPU.
    """

    def __init__(
        self,
        model_cfg: str,
        model_weights: str,
        class_names: str,
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        input_size: int = 416,
        device: str = "auto",
        person_only: bool = True,
    ) -> None:
        for path, label in (
            (model_cfg, "model config"),
            (model_weights, "model weights"),
            (class_names, "class names"),
        ):
            if not os.path.isfile(path):
                raise DetectorError(
                    f"Missing {label} file: '{path}'. "
                    "Verify the --model path or re-download the bundled weights "
                    "(see data/README.md)."
                )

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.person_only = person_only

        try:
            self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        except cv2.error as exc:
            raise DetectorError(
                f"Failed to load detection model from '{model_cfg}' / "
                f"'{model_weights}': {exc}"
            ) from exc

        self.device_used = self._configure_device(device)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        with open(class_names, "r", encoding="utf-8") as f:
            self.class_names: List[str] = [line.strip() for line in f if line.strip()]

        logger.info(
            "PersonDetector initialized (device=%s, input_size=%d, "
            "conf_thresh=%.2f, nms_thresh=%.2f)",
            self.device_used,
            self.input_size,
            self.confidence_threshold,
            self.nms_threshold,
        )

    def _configure_device(self, device: str) -> str:
        """Select CPU or CUDA backend for inference, degrading gracefully."""
        device = device.lower()
        cuda_build = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False

        if device == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return "cpu"

        if device in ("cuda", "gpu", "auto"):
            if cuda_build:
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    return "cuda"
                except cv2.error as exc:
                    logger.warning(
                        "CUDA backend requested but failed to initialize (%s). "
                        "Falling back to CPU.",
                        exc,
                    )
            elif device in ("cuda", "gpu"):
                logger.warning(
                    "CUDA device requested but this OpenCV build has no CUDA "
                    "support (or no GPU is available). Falling back to CPU. "
                    "Install an opencv-contrib build with CUDA support to enable GPU inference."
                )
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return "cpu"

        raise DetectorError(f"Unknown device option '{device}'. Use auto|cpu|cuda.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run person detection on a single BGR frame.

        Returns an empty list (never raises) for frames with no detections,
        since "no detections" is an expected, normal outcome, not an error.
        """
        if frame is None or frame.size == 0:
            logger.debug("detect() called with an empty frame; skipping.")
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        try:
            outputs = self.net.forward(self.output_layers)
        except cv2.error as exc:
            logger.error("Inference failed on this frame, skipping it: %s", exc)
            return []

        boxes: List[List[float]] = []
        confidences: List[float] = []

        for output in outputs:
            for det in output:
                class_scores = det[5:]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])

                if self.person_only and class_id != COCO_PERSON_CLASS_ID:
                    continue
                if confidence < self.confidence_threshold:
                    continue

                cx, cy, bw, bh = det[0] * w, det[1] * h, det[2] * w, det[3] * h
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        if len(indices) == 0:
            return []

        detections: List[Detection] = []
        for i in np.array(indices).flatten():
            x1, y1, bw, bh = boxes[i]
            x2, y2 = x1 + bw, y1 + bh
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=confidences[i]))

        return detections
