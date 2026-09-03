"""
pipeline.py
-----------
Orchestrates the end-to-end surveillance pipeline: video I/O, detection,
tracking, zone/event logic, annotation, and output generation.

Frames are read and processed one at a time (a generator-driven loop) so
memory usage stays flat regardless of video length -- the entire video is
never loaded into memory, only the current frame and a small amount of
per-track state.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, DetectorError, PersonDetector
from .events import Event, EventManager
from .logger import EventLogWriter
from .tracker import PersonTracker, Track
from .utils import ensure_dir, frame_to_timestamp
from .zone_manager import ZoneManager

logger = logging.getLogger("surveillance")

# BGR colors
COLOR_TEXT = (255, 255, 255)
COLOR_BOX = (0, 220, 0)
COLOR_BOX_ALERT = (0, 0, 255)
COLOR_HUD_BG = (20, 20, 20)


class PipelineError(RuntimeError):
    """Raised for unrecoverable pipeline configuration/runtime errors."""


@dataclass
class PipelineStats:
    frames_processed: int = 0
    frames_with_detections: int = 0
    total_detections: int = 0
    intrusion_events: int = 0
    loitering_events: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    video_fps: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return max(end - self.start_time, 1e-6)

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.elapsed_seconds


class SurveillancePipeline:
    """Runs detection + tracking + zone/event logic over a video file."""

    def __init__(
        self,
        detector: PersonDetector,
        tracker: PersonTracker,
        zone_manager: ZoneManager,
        output_dir: str,
        save_video: bool = True,
        save_events: bool = True,
        event_formats: Tuple[str, ...] = ("json", "csv"),
        fps_log_interval: int = 100,
        recent_alert_seconds: float = 3.0,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.zone_manager = zone_manager
        self.output_dir = output_dir
        self.save_video = save_video
        self.save_events = save_events
        self.event_formats = event_formats
        self.fps_log_interval = fps_log_interval
        self.recent_alert_seconds = recent_alert_seconds

        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "annotated"))
        ensure_dir(os.path.join(output_dir, "events"))

        self.stats = PipelineStats()
        self._recent_alerts: List[Tuple[float, str]] = []  # (video_time, message)

    def run(self, video_path: str) -> PipelineStats:
        if not os.path.isfile(video_path):
            raise PipelineError(
                f"Input video not found: '{video_path}'. Check the --video path."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise PipelineError(
                f"Could not open video '{video_path}'. The file may be corrupt, "
                "use an unsupported codec, or not be a video file at all."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            logger.warning(
                "Video reports an invalid FPS (%s); defaulting to 25.0. "
                "Timestamps may be inaccurate.",
                fps,
            )
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise PipelineError(
                f"Video '{video_path}' reported invalid dimensions ({width}x{height}). "
                "The file may be corrupt or use an unsupported codec."
            )

        logger.info(
            "Opened video '%s': %dx%d @ %.2f fps, ~%d frames",
            video_path, width, height, fps, total_frames,
        )

        self.stats.video_fps = fps
        event_manager = EventManager(fps=fps)
        event_writer = EventLogWriter(os.path.join(self.output_dir, "events"))

        writer = None
        annotated_path = None
        if self.save_video:
            annotated_path = os.path.join(
                self.output_dir, "annotated", self._output_filename(video_path)
            )
            writer = self._open_video_writer(annotated_path, fps, width, height)

        self.stats = PipelineStats(video_fps=fps)
        frame_number = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break  # end of stream (also covers unreadable/corrupt frame)
                frame_number += 1
                self.stats.frames_processed += 1

                detections = self._safe_detect(frame)
                if detections:
                    self.stats.frames_with_detections += 1
                    self.stats.total_detections += len(detections)

                tracks = self.tracker.update(detections)
                new_events = self._process_zones_and_events(
                    frame_number, tracks, event_manager
                )
                for ev in new_events:
                    if ev.event_type == "zone_intrusion":
                        self.stats.intrusion_events += 1
                    elif ev.event_type == "loitering":
                        self.stats.loitering_events += 1
                    self._register_alert(frame_number, fps, ev)

                if writer is not None:
                    annotated = self._annotate_frame(
                        frame, tracks, frame_number, fps, new_events
                    )
                    writer.write(annotated)

                if self.fps_log_interval and frame_number % self.fps_log_interval == 0:
                    logger.info(
                        "Processed %d frame(s) | running processing FPS: %.1f | "
                        "events so far: %d",
                        frame_number,
                        frame_number / max(time.time() - self.stats.start_time, 1e-6),
                        len(event_manager.events),
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        self.stats.end_time = time.time()

        if self.save_events:
            if "json" in self.event_formats:
                event_writer.write_json(event_manager.events)
            if "csv" in self.event_formats:
                event_writer.write_csv(event_manager.events)

        if self.stats.frames_processed == 0:
            logger.warning(
                "The video produced zero readable frames. No output was generated."
            )

        self._log_summary(annotated_path if self.save_video else None)
        return self.stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_detect(self, frame: np.ndarray) -> List[Detection]:
        try:
            return self.detector.detect(frame)
        except DetectorError as exc:
            logger.error("Detector error on this frame, skipping: %s", exc)
            return []

    def _process_zones_and_events(
        self, frame_number: int, tracks: List[Track], event_manager: EventManager
    ) -> List[Event]:
        tracks_in_zones = []
        for t in tracks:
            zones = self.zone_manager.zones_containing(t.bbox)
            tracks_in_zones.append((t.track_id, t.bbox, t.confidence, [z.name for z in zones]))

        zone_lookup = {z.name: z for z in self.zone_manager.zones}
        new_events = event_manager.process_frame(frame_number, tracks_in_zones, zone_lookup)

        active_ids = [t.track_id for t in self.tracker.tracks.values()]
        event_manager.handle_lost_tracks(active_ids)
        return new_events

    def _register_alert(self, frame_number: int, fps: float, event: Event) -> None:
        video_time = frame_number / fps
        if event.event_type == "zone_intrusion":
            msg = f"INTRUSION: track {event.track_id} entered '{event.zone}'"
        else:
            msg = (
                f"LOITERING: track {event.track_id} in '{event.zone}' "
                f"for {event.duration_seconds:.1f}s"
            )
        self._recent_alerts.append((video_time, msg))
        logger.info("EVENT %s | frame %d | %s", event.event_id, frame_number, msg)

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_number: int,
        fps: float,
        new_events: List[Event],
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw zones first so boxes/labels render on top.
        for zone in self.zone_manager.zones:
            pts = zone.as_np().reshape(-1, 1, 2)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], zone.color)
            cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, dst=annotated)
            cv2.polylines(annotated, [pts], isClosed=True, color=zone.color, thickness=2)
            label_pos = (int(zone.polygon[0][0]), max(int(zone.polygon[0][1]) - 8, 15))
            cv2.putText(
                annotated, zone.name, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone.color, 2, cv2.LINE_AA,
            )

        # Track boxes + IDs + confidence.
        alerted_ids = {e.track_id for e in new_events}
        for t in tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in t.bbox)
            color = COLOR_BOX_ALERT if t.track_id in alerted_ids else COLOR_BOX
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {t.track_id} | {t.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, max(y1 - 5, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        self._draw_hud(annotated, frame_number, fps, len(tracks))
        self._draw_alert_banner(annotated, frame_number, fps)
        return annotated

    def _draw_hud(self, frame: np.ndarray, frame_number: int, fps: float, n_tracks: int) -> None:
        h, w = frame.shape[:2]
        elapsed = time.time() - self.stats.start_time
        proc_fps = frame_number / max(elapsed, 1e-6)
        timestamp = frame_to_timestamp(frame_number, fps)

        lines = [
            f"Frame {frame_number} | t={timestamp} | tracks={n_tracks}",
            f"Processing FPS: {proc_fps:.1f}",
        ]
        pad = 6
        line_h = 20
        box_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for l in lines) + 2 * pad
        box_h = line_h * len(lines) + pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h - 6
            cv2.putText(
                frame, line, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
            )

    def _draw_alert_banner(self, frame: np.ndarray, frame_number: int, fps: float) -> None:
        video_time = frame_number / fps
        self._recent_alerts = [
            (t, m) for t, m in self._recent_alerts
            if video_time - t <= self.recent_alert_seconds
        ]
        if not self._recent_alerts:
            return
        h, w = frame.shape[:2]
        msg = self._recent_alerts[-1][1]
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = h - th - 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0 - 10), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
        cv2.putText(
            frame, msg, (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _open_video_writer(self, path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
        # mp4v is the most broadly available OpenCV FourCC across platforms
        # without extra codec packages; documented in README as the codec
        # assumption. Falls back to a .avi/MJPG container if mp4 fails to
        # open (e.g. missing codec on a minimal Linux install).
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning(
                "Could not open video writer with 'mp4v' codec for '%s'. "
                "Falling back to MJPG/.avi output.",
                path,
            )
            fallback_path = os.path.splitext(path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(fallback_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise PipelineError(
                    f"Failed to open a video writer for output path '{path}' "
                    "with both mp4v and MJPG codecs. Check that OpenCV was "
                    "built with video-writing support."
                )
        return writer

    @staticmethod
    def _output_filename(video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        return f"{base}_annotated.mp4"

    def _log_summary(self, annotated_path: Optional[str]) -> None:
        s = self.stats
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("Video FPS: %.2f", s.video_fps)
        logger.info("Processing FPS: %.2f", s.processing_fps)
        logger.info("Frames processed: %d", s.frames_processed)
        logger.info("Frames with detections: %d", s.frames_with_detections)
        logger.info("Total detections: %d", s.total_detections)
        logger.info("Total processing time: %.2fs", s.elapsed_seconds)
        logger.info(
            "Events detected: %d (intrusion=%d, loitering=%d)",
            s.intrusion_events + s.loitering_events,
            s.intrusion_events,
            s.loitering_events,
        )
        if annotated_path:
            logger.info("Annotated video: %s", annotated_path)
        logger.info("=" * 60)
