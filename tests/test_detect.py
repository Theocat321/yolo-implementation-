import numpy as np
from detect import format_label, draw_fps


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
