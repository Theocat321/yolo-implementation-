import numpy as np
from detect import format_label, draw_fps, draw_detections


def test_format_label_rounds_to_integer_percent():
    assert format_label("person", 0.874) == "person (87%)"


def test_format_label_rounds_up():
    assert format_label("car", 0.999) == "car (100%)"


def test_format_label_low_confidence():
    assert format_label("dog", 0.501) == "dog (50%)"


def test_draw_fps_returns_same_shape_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_fps(frame, 30.0)
    assert result.shape == (480, 640, 3)


def test_draw_fps_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_fps(frame, 30.0)
    assert result.any(), "Expected draw_fps to write at least one non-zero pixel"


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
