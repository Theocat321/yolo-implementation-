# YOLOv11 Webcam Object Detection — Design Spec

**Date:** 2026-05-19  
**Status:** Approved

## Overview

A single Python script (`detect.py`) that opens the default webcam, runs YOLOv11 nano inference on each frame, and displays the annotated feed in an OpenCV window. Press `q` to quit.

## Architecture

Single script, no classes. Three logical sections: setup, frame loop, teardown.

## Components & Data Flow

1. **Webcam capture** — `cv2.VideoCapture(0)` opens the default webcam
2. **Model** — `YOLO("yolo11n.pt")` loads YOLOv11 nano (auto-downloads weights on first run); nano chosen for real-time performance
3. **Frame loop:**
   - Read frame from capture
   - Run `model(frame, verbose=False)` → results with boxes, class IDs, confidences
   - For each detection above the confidence threshold: draw `cv2.rectangle` and `cv2.putText` with `label (conf%)`
   - Draw FPS counter in top-left corner
   - `cv2.imshow` to display the annotated frame
   - Break on `q` keypress
4. **Teardown** — release capture, destroy all windows

## Configuration Constants

| Constant | Default | Purpose |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to show a detection |
| `WEBCAM_INDEX` | `0` | Which camera to use |

## Detection

- All 80 COCO classes detected (person, car, dog, bottle, etc.)
- Each box labeled with class name and confidence percentage
- Single color for all boxes (green) for simplicity

## Dependencies

```
ultralytics
opencv-python
```

`ultralytics` bundles PyTorch and downloads model weights automatically on first run.

## Success Criteria

- Webcam feed opens in a window with bounding boxes and labels
- Runs at usable frame rate on CPU (target: >10 FPS with yolo11n)
- `q` key cleanly exits
