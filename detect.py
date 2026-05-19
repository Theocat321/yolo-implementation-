import time
import cv2
import numpy as np
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5
WEBCAM_INDEX = 0


def format_label(class_name: str, confidence: float) -> str:
    return f"{class_name} ({round(confidence * 100)}%)"


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
