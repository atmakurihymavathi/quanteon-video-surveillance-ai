#!/usr/bin/env python3
"""
generate_synthetic_video.py
----------------------------
Generates a small synthetic MP4 clip with moving humanoid silhouettes that
walk into, loiter inside, and leave the zones defined in
config/zones.json. This gives a fully deterministic, dependency-free
fixture for exercising the zone/event/pipeline logic in tests and demos
without requiring a real detector or downloaded footage.

This is a pipeline-integration fixture, not a detection-accuracy
benchmark -- run the real detector against genuine footage (see
data/README.md) to evaluate detection quality.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def draw_person(frame: np.ndarray, cx: int, cy: int, scale: float = 1.0) -> None:
    """Draw a simple filled humanoid silhouette centered at (cx, cy) (feet position)."""
    color = (40, 40, 40)
    head_r = int(10 * scale)
    body_h = int(45 * scale)
    body_w = int(18 * scale)
    cv2.circle(frame, (cx, cy - body_h - head_r), head_r, color, -1)
    cv2.rectangle(
        frame,
        (cx - body_w // 2, cy - body_h),
        (cx + body_w // 2, cy),
        color,
        -1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic_demo.mp4")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    n_frames = args.fps * args.seconds

    # Person A: walks straight through the restricted_area zone (brief intrusion).
    # Person B: walks into restricted_area and stops (loitering).
    # Person C: walks across empty space only (no zone interaction, negative case).
    for i in range(n_frames):
        frame = np.full((args.height, args.width, 3), 235, dtype=np.uint8)
        cv2.putText(
            frame, f"synthetic frame {i}", (10, args.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
        )

        # Person A: fast horizontal walk through the top of restricted_area.
        ax = int(50 + (i / n_frames) * (args.width - 100))
        draw_person(frame, ax, 150)

        # Person B: walks in during first third, then stands still (loiters).
        enter_frames = n_frames // 3
        if i < enter_frames:
            bx = int(120 + (i / enter_frames) * 180)
        else:
            bx = 300
        draw_person(frame, bx, 300, scale=1.1)

        # Person C: walks along the bottom, outside any configured zone.
        cx = int(20 + (i / n_frames) * (args.width - 40))
        draw_person(frame, cx, args.height - 30, scale=0.9)

        writer.write(frame)

    writer.release()
    print(f"Wrote {n_frames} frames ({args.seconds}s @ {args.fps}fps) to {args.output}")


if __name__ == "__main__":
    main()
