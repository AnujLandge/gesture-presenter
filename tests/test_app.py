"""Tests for the drawing helpers in app.py (no camera or window needed)."""
import numpy as np

from gesture_presenter import app


def test_smooth_stroke_keeps_ends_and_reduces_zigzag():
    """Smoothing keeps the first and last point and flattens a zigzag line."""
    zigzag = [(i * 10, 100 + (8 if i % 2 else -8)) for i in range(20)]
    smoothed = app.smooth_stroke(zigzag)
    assert smoothed[0] == zigzag[0]
    assert smoothed[-1] == zigzag[-1]
    assert len(smoothed) == len(zigzag)
    middle = smoothed[3:-3]
    assert max(abs(y - 100) for _, y in middle) <= 2


def test_short_stroke_is_left_alone():
    """A stroke with only a few points is returned unchanged."""
    dot = [(5, 5), (6, 6)]
    assert app.smooth_stroke(dot) == dot


def test_fit_to_window_pads_a_taller_window(monkeypatch):
    """A 16:10 window gets equal padding above and below, in the slide's edge colour."""
    monkeypatch.setattr(app.cv2, "getWindowImageRect", lambda name: (0, 0, 1600, 1000))
    slide = np.full((app.H, app.W, 3), (140, 42, 75), np.uint8)
    out = app.fit_to_window(slide.copy(), slide)
    assert out.shape == (1200, app.W, 3)
    assert tuple(out[0, 0]) == (140, 42, 75)
    assert tuple(out[-1, -1]) == (140, 42, 75)


def test_fit_to_window_leaves_a_16_9_window_alone(monkeypatch):
    """A window that already has the slide's shape needs no padding."""
    monkeypatch.setattr(app.cv2, "getWindowImageRect", lambda name: (0, 0, 1280, 720))
    slide = np.zeros((app.H, app.W, 3), np.uint8)
    assert app.fit_to_window(slide, slide).shape == (app.H, app.W, 3)
