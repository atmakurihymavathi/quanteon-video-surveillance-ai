# Model Weights

This directory holds the pretrained detector files used by `src/detector.py`:

```
models/
├── yolov4-tiny.cfg       # Darknet network architecture (text, ~12KB)
├── yolov4-tiny.weights   # Pretrained COCO weights (~23MB)
└── coco.names            # 80 COCO class names (person = class 0)
```

These are the **official pretrained YOLOv4-tiny weights**, trained on the
COCO dataset by the Darknet/YOLOv4 authors. No training or fine-tuning is
performed or required by this project -- see the README.md "Model
Selection" section for why a pretrained, off-the-shelf detector is the
right choice for this assignment's scope.

## Re-downloading the weights

`yolov4-tiny.weights` is excluded from version control (see `.gitignore`)
because binary model weights don't belong in a git history. If it's
missing, fetch it with:

```bash
mkdir -p models
curl -L -o models/yolov4-tiny.cfg \
  https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
curl -L -o models/coco.names \
  https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
curl -L -o models/yolov4-tiny.weights \
  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

`run.py` will raise a clear `DetectorError` (not a cryptic stack trace) if
any of these three files are missing when the pipeline starts.

## Using a different model

`PersonDetector` loads any Darknet-format `.cfg` + `.weights` pair, so you
can swap in a different YOLO variant (e.g. full YOLOv4, YOLOv3) via:

```bash
python run.py --video ... --zones ... --output ... --model models/yolov4
# expects models/yolov4.cfg and models/yolov4.weights
```
