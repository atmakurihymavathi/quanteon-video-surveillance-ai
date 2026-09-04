# Video Surveillance AI — Detection, Tracking & Event Recognition

**Quanteon Solutions AI Engineer Take-Home Assignment**

A CPU-friendly, dependency-light computer vision pipeline that detects
people in security camera footage, tracks them across frames with persistent
IDs, and raises timestamped **zone intrusion** and **loitering** events against
configurable polygon zones.

The system produces annotated video output and structured JSON/CSV event logs,
with configurable detection, tracking, event, and runtime parameters.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Features](#2-features)
3. [Architecture Overview](#3-architecture-overview)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Pipeline Flow](#5-pipeline-flow)
6. [Model Selection](#6-model-selection)
7. [Detector](#7-detector)
8. [Tracker](#8-tracker)
9. [Re-Identification Approach](#9-re-identification-approach)
10. [Zone Detection Logic](#10-zone-detection-logic)
11. [Intrusion Logic](#11-intrusion-logic)
12. [Loitering Logic](#12-loitering-logic)
13. [Event Deduplication](#13-event-deduplication)
14. [Configuration](#14-configuration)
15. [Installation](#15-installation)
16. [Requirements](#16-requirements)
17. [Usage / CLI Examples](#17-usage--cli-examples)
18. [Sample Output](#18-sample-output)
19. [Event Log Examples](#19-event-log-examples)
20. [Performance](#20-performance)
21. [CPU / GPU Notes](#21-cpu--gpu-notes)
22. [Edge Cases](#22-edge-cases)
23. [Known Limitations](#23-known-limitations)
24. [Future Improvements](#24-future-improvements)
25. [Project Structure](#25-project-structure)
26. [Testing](#26-testing)
27. [Reproducibility](#27-reproducibility)
28. [Dataset / Video Sources](#28-dataset--video-sources)

---

## 1. Problem Statement

The objective is to build a system that processes security camera footage to:

- Detect people in video frames
- Track detected people across frames using persistent IDs
- Detect security events such as **zone intrusion** and **loitering**
- Produce timestamped event records
- Generate an annotated video showing detections, tracking IDs, zones, and alerts
- Provide configurable and production-minded engineering rather than a research-only prototype

The implementation focuses on modularity, error handling, configuration,
reproducibility, and practical CPU performance.

---

## 2. Features

- Person detection with bounding boxes and confidence scores using **YOLOv4-tiny**
- Multi-object tracking with persistent track IDs
- Short-term track continuity through brief occlusions or missed detections
- Polygon-based, JSON-configurable zones
- Support for multiple zones, including convex and concave polygons
- Zone intrusion detection using edge-triggered state transitions
- Loitering detection with configurable per-zone thresholds
- Event deduplication to avoid per-frame event flooding
- Timestamped structured **JSON and CSV event logs**
- Annotated output video with:
  - Bounding boxes
  - Track IDs
  - Detection confidence
  - Polygon zones
  - FPS
  - Frame/timestamp information
  - Intrusion/loitering alerts
- CLI with configurable detection, tracking, event, output, and device options
- CPU-first execution with graceful CUDA fallback
- Streaming frame-by-frame processing
- Multi-video processing support through repeated `--video` arguments
- FPS benchmarking
- MOT-format evaluation scaffold using `motmetrics`
- **60 unit tests** covering geometry, tracking, event state machines,
  configuration, and evaluation utilities

---

## 3. Architecture Overview

The pipeline is designed as a modular, single-pass frame processor:

```text
Video
  |
  v
[Video Reader]
  |
  v
[Person Detector]
  |
  v
[Person Tracker]
  |
  v
[Zone Manager]
  |
  v
[Event Manager]
  |
  +--------------------+
  |                    |
  v                    v
[Annotator]       [Event Logger]
  |                    |
  v                    v
Annotated Video    JSON / CSV
