# yolo-implementation-

Real-time object detection using YOLOv11 nano and your webcam. Displays a live feed with bounding boxes, class labels, confidence scores, and an FPS counter.

## Requirements

- Python 3.8+
- Webcam

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python detect.py
```

On first run, YOLOv11 nano weights (~6 MB) are downloaded automatically. A window opens showing the annotated webcam feed. Press `q` to quit.

## What it detects

All 80 COCO classes — person, car, dog, bottle, and more. Detections below 50% confidence are filtered out.

## Tests

```bash
pytest tests/ -v
```
