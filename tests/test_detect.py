from detect import format_label


def test_format_label_rounds_to_integer_percent():
    assert format_label("person", 0.874) == "person (87%)"


def test_format_label_rounds_up():
    assert format_label("car", 0.999) == "car (100%)"


def test_format_label_low_confidence():
    assert format_label("dog", 0.501) == "dog (50%)"
