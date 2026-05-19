import time
import cv2
import numpy as np
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5
WEBCAM_INDEX = 0


def format_label(class_name: str, confidence: float) -> str:
    return f"{class_name} ({round(confidence * 100)}%)"
