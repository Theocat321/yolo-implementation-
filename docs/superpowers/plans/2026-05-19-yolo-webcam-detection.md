# YOLOv11 Webcam Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Python script that opens the default webcam, runs YOLOv11 nano inference on every frame, and displays bounding boxes with labels in an OpenCV window.

**Architecture:** `detect.py` exposes three pure helper functions (`format_label`, `draw_fps`, `draw_detections`) that are unit-tested independently, and a `main()` function that wires together the webcam loop. Pure functions take plain numpy arrays so they have no dependency on ultralytics internals and are easy to test.

**Tech Stack:** Python 3.8+, ultralytics (YOLOv11), opencv-python, pytest

---

## File Structure

| Path | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `detect.py` | Pure helper functions + `main()` entry point |
| `tests/__init__.py` | Empty — marks tests as a package |
| `tests/test_detect.py` | Unit tests for `format_label`, `draw_fps`, `draw_detections` |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
ultralytics
opencv-python
pytest
```

- [ ] **Step 2: Create empty tests package**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: packages install without errors. `ultralytics` will pull in torch; this may take a minute.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "chore: add dependencies and test package"
```

---

## Task 2: Label Formatter

**Files:**
- Create: `detect.py` (initial, contains only `format_label`)
- Create: `tests/test_detect.py` (initial, tests `format_label`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_detect.py`:

```python
from detect import format_label


def test_format_label_rounds_to_integer_percent():
    assert format_label("person", 0.874) == "person (87%)"


def test_format_label_rounds_up():
    assert format_label("car", 0.999) == "car (100%)"


def test_format_label_low_confidence():
    assert format_label("dog", 0.501) == "dog (50%)"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_detect.py -v
```

Expected: `ModuleNotFoundError: No module named 'detect'`

- [ ] **Step 3: Create detect.py with format_label**

```python
import time
import cv2
import numpy as np
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5
WEBCAM_INDEX = 0


def format_label(class_name: str, confidence: float) -> str:
    return f"{class_name} ({int(confidence * 100)}%)"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_detect.py -v
```

Expected:
```
tests/test_detect.py::test_format_label_rounds_to_integer_percent PASSED
tests/test_detect.py::test_format_label_rounds_up PASSED
tests/test_detect.py::test_format_label_low_confidence PASSED
```

- [ ] **Step 5: Commit**

```bash
git add detect.py tests/test_detect.py
git commit -m "feat: add format_label and tests"
```

---

## Task 3: FPS Counter Drawing

**Files:**
- Modify: `detect.py` — add `draw_fps`
- Modify: `tests/test_detect.py` — add FPS tests

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detect.py`:

```python
def test_draw_fps_returns_same_shape_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_fps(frame, 30.0)
    assert result.shape == (480, 640, 3)


def test_draw_fps_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_fps(frame, 30.0)
    assert result.any(), "Expected draw_fps to write at least one non-zero pixel"
```

Update the import at the top of `tests/test_detect.py`:

```python
from detect import format_label, draw_fps
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_detect.py::test_draw_fps_returns_same_shape_frame -v
```

Expected: `ImportError: cannot import name 'draw_fps'`

- [ ] **Step 3: Add draw_fps to detect.py**

Append after `format_label`:

```python
def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_detect.py -v
```

Expected: all 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add detect.py tests/test_detect.py
git commit -m "feat: add draw_fps and tests"
```

---

## Task 4: Detection Drawing

**Files:**
- Modify: `detect.py` — add `draw_detections`
- Modify: `tests/test_detect.py` — add detection drawing tests

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detect.py`:

```python
def test_draw_detections_returns_same_shape_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = np.array([[50, 50, 200, 200]], dtype=np.float32)
    class_ids = np.array([0], dtype=np.float32)
    confidences = np.array([0.9], dtype=np.float32)
    names = {0: "person"}
    result = draw_detections(frame, boxes, class_ids, confidences, names, threshold=0.5)
    assert result.shape == (480, 640, 3)


def test_draw_detections_skips_low_confidence():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = np.array([[50, 50, 200, 200]], dtype=np.float32)
    class_ids = np.array([0], dtype=np.float32)
    confidences = np.array([0.3], dtype=np.float32)  # below threshold
    names = {0: "person"}
    result = draw_detections(frame, boxes, class_ids, confidences, names, threshold=0.5)
    assert not result.any(), "Expected no pixels drawn for low-confidence detection"


def test_draw_detections_draws_above_threshold():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = np.array([[50, 50, 200, 200]], dtype=np.float32)
    class_ids = np.array([0], dtype=np.float32)
    confidences = np.array([0.9], dtype=np.float32)
    names = {0: "person"}
    result = draw_detections(frame, boxes, class_ids, confidences, names, threshold=0.5)
    assert result.any(), "Expected pixels drawn for high-confidence detection"
```

Update the import at the top of `tests/test_detect.py`:

```python
from detect import format_label, draw_fps, draw_detections
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_detect.py::test_draw_detections_returns_same_shape_frame -v
```

Expected: `ImportError: cannot import name 'draw_detections'`

- [ ] **Step 3: Add draw_detections to detect.py**

Append after `draw_fps`:

```python
def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    names: dict,
    threshold: float = 0.5,
) -> np.ndarray:
    for box, class_id, conf in zip(boxes_xyxy, class_ids, confidences):
        if conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = format_label(names[int(class_id)], conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_detect.py -v
```

Expected: all 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add detect.py tests/test_detect.py
git commit -m "feat: add draw_detections and tests"
```

---

## Task 5: Main Loop

**Files:**
- Modify: `detect.py` — add `main()` and `__main__` guard

- [ ] **Step 1: Append main() to detect.py**

```python
def main() -> None:
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {WEBCAM_INDEX}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame — exiting.")
            break

        results = model(frame, verbose=False)

        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            draw_detections(
                frame,
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                model.names,
                CONFIDENCE_THRESHOLD,
            )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        draw_fps(frame, fps)

        cv2.imshow("YOLOv11 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to confirm nothing broke**

```bash
pytest tests/ -v
```

Expected: all 8 tests PASSED.

- [ ] **Step 3: Manual smoke test**

```bash
python detect.py
```

Expected:
- A window titled "YOLOv11 Detection" opens showing the webcam feed
- On first run, `yolo11n.pt` downloads automatically (~6 MB)
- Bounding boxes with green labels appear around detected objects
- FPS counter shown in top-left
- Press `q` to close cleanly

- [ ] **Step 4: Commit**

```bash
git add detect.py
git commit -m "feat: add main loop — webcam + YOLOv11 inference complete"
```
