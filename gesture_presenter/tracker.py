"""Hand tracking: step 2 of the pipeline (camera image -> 21 hand landmarks).

A thin wrapper around Google's MediaPipe Tasks HandLandmarker (mediapipe >=
0.10.x). We use the pre-trained model as-is; this file only finds the model
file, runs it on each frame and returns plain (x, y) tuples so the rest of the
program does not depend on MediaPipe types.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# Where the pre-trained hand model is downloaded from on first run (~7 MB).
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_FILENAME = "hand_landmarker.task"

# Folder holding this file, or PyInstaller's unpack folder (sys._MEIPASS)
# when running as the packaged macOS app.
_BUNDLE_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
# Model shipped inside the packaged app (see build_mac.sh --add-data).
_BUNDLED = os.path.join(_BUNDLE_ROOT, "gesture_presenter", "models", MODEL_FILENAME)
# Model placed next to the source code in gesture_presenter/models/.
_LOCAL = os.path.join(os.path.dirname(__file__), "models", MODEL_FILENAME)
# Download location used when neither of the above exists.
_CACHED = os.path.join(os.path.expanduser("~"), ".cache", "gesture-presenter", MODEL_FILENAME)

MAX_HANDS = 1                  # we only ever follow one hand
MIN_CONFIDENCE = 0.5           # detection / presence confidence needed to report a hand
MIN_TRACKING_CONFIDENCE = 0.5  # below this MediaPipe re-detects instead of tracking


def _find_model_path() -> str:
    """Return the first model file that exists, else the download location."""
    for candidate in (_BUNDLED, _LOCAL):
        if os.path.exists(candidate):
            return candidate
    return _CACHED


MODEL_PATH = _find_model_path()


def ensure_model(path: str = MODEL_PATH) -> str:
    """Download the hand model to `path` if it is not there yet; return the path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading hand model (~7 MB, one time)...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Finds one hand per frame and returns its 21 landmarks."""

    def __init__(self, min_conf: float = MIN_CONFIDENCE):
        """Load the model in VIDEO mode, which tracks the hand between frames."""
        # VIDEO mode reuses the previous frame's result, which is faster and
        # steadier than detecting the hand from scratch on every frame.
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        """Milliseconds since start, always larger than the previous value."""
        timestamp = int((time.monotonic() - self.start_time) * 1000)
        # MediaPipe VIDEO mode rejects a timestamp that does not strictly
        # increase, which can happen if two frames arrive in the same millisecond.
        timestamp = max(timestamp, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp
        return timestamp

    def process(self, rgb):
        """rgb: HxWx3 uint8 RGB frame. Returns list of 21 (x, y) or None."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, self._next_timestamp_ms())
        if not result.hand_landmarks:
            return None
        first_hand = result.hand_landmarks[0]
        points = []
        for landmark in first_hand:
            points.append((landmark.x, landmark.y))
        return points

    def close(self):
        """Release the MediaPipe model."""
        self.detector.close()
