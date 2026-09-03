# Sample Video Data

This directory is where you should place input video files for the
pipeline (e.g. `data/sample.mp4`). Video files are **not** committed to
this repository (see `.gitignore`) both to keep the repository small and
to avoid redistributing footage under uncertain licensing terms.

## Where to get a video

**Option A -- use your own footage.** Any MP4/AVI/MOV file with people in
frame works. This is the fastest way to try the pipeline end-to-end.

**Option B -- use one of the public datasets suggested in the assignment:**

| Dataset | Contents | Source |
|---|---|---|
| MOT17 | Pedestrian-focused tracking benchmark with ground-truth boxes/IDs. Best for evaluating tracking accuracy with `src/evaluation.py`. | https://motchallenge.net/data/MOT17/ |
| VIRAT | Outdoor pedestrian/vehicle surveillance, multiple resolutions. | https://viratdata.org/ |
| UCF-Crime | Real-world CCTV footage incl. a `Loitering` category. | https://www.crcv.ucf.edu/projects/real-world/ (Kaggle mirror also available) |
| MEVA / MEVADATA | Large-scale multi-camera ground + UAV footage. | https://mevadata.org/ |
| VisDrone | Drone-captured pedestrian/vehicle sequences. | https://github.com/VisDrone/VisDrone-Dataset |

Download 1-2 short (30-60s) clips, place them here (e.g. `data/mot17_clip1.mp4`),
and point `--video` at the file.

## Included synthetic test clip

Because this repository is built and evaluated in a sandboxed environment
without general internet access, it cannot download footage from the
sites above at build time. To still exercise and verify the *entire*
pipeline end-to-end (detection through to zone events), a small
synthetic clip generator is provided:

```bash
python scripts/generate_synthetic_video.py --output data/synthetic_demo.mp4
```

This draws simple moving humanoid silhouettes that walk into and loiter
inside the configured zones, so intrusion and loitering logic can be
verified deterministically without a real detector even being present.
It is a **pipeline-integration smoke test**, not a substitute for
evaluating detection accuracy on real footage -- run the pipeline on a
real MOT17/VIRAT/UCF-Crime clip for a meaningful accuracy assessment (see
README.md "Known Limitations").
