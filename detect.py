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


def main() -> None:
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {WEBCAM_INDEX}")

    prev_time = time.time()

    try:
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
            elapsed = curr_time - prev_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            prev_time = curr_time
            draw_fps(frame, fps)

            cv2.imshow("YOLOv11 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
