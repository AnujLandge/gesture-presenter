"""Hand Gesture Presentation System - main loop that ties the pipeline together.

Usage:
  python -m gesture_presenter deck.pptx     # .pptx .ppt .pps(x) .odp .key .pdf
  python -m gesture_presenter               # opens a file picker
Keys: q/Esc quit, f fullscreen, c camera preview, h help panel, a/d or arrows manual nav.

Every frame goes through the same steps:
  1. take the newest webcam image (Camera, on its own thread)
  2. find the hand's 21 landmarks (tracker.HandTracker)
  3. turn landmarks into a steady pose and maybe an action (gestures.GestureEngine)
  4. apply the action to the slides (Presenter.act)
  5. move the pointer / extend the drawing (update_pointer)
  6. draw the slide, drawings, pointer and HUD (Presenter.render, overlay)
  7. show the picture and handle any key press (handle_key)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from . import ui
from .gestures import CONTINUOUS, Gesture, GestureEngine, Smoother, INDEX_TIP
from . import slides as deck
from .tracker import HandTracker

# --- Output canvas ------------------------------------------------------------
W, H = 1920, 1080                       # every slide is drawn on a canvas this size
WIN = "Hand Gesture Presenter"          # window title
WINDOW_START_SIZE = (1280, 720)         # window size before going fullscreen

# --- Camera -------------------------------------------------------------------
CAMERA_SIZE = (960, 540)                # resolution requested from the webcam
CAMERA_FPS = 30                         # frame rate requested from the webcam
TRACKING_SIZE = (640, 360)              # image size given to MediaPipe (smaller = faster)
CAMERA_PERMISSION_WAIT_S = 30.0         # how long to keep retrying while macOS asks for camera access
CAMERA_RETRY_S = 1.0                    # pause between camera open attempts
FRAME_TIMEOUT_S = 1.0                   # stop if the camera sends nothing for this long
CAMERA_THREAD_IDLE_S = 0.01             # pause after a failed camera read
FRAME_POLL_S = 0.001                    # how often the main loop checks for a new frame
CAMERA_STOP_WAIT_S = 0.05               # time given to the camera thread to finish

# --- Timings ------------------------------------------------------------------
HELP_AUTO_HIDE_S = 15                   # help card hides itself after this; press h to bring it back
TOAST_DURATION_S = 0.9                  # how long the action toast stays on screen

# --- Drawing ------------------------------------------------------------------
STROKE_COLOUR = (40, 40, 225)           # red ink (BGR)
STROKE_WIDTH = 4                        # pen width in design points (scaled by ui.px)
MIN_POINT_GAP_PX = 3                    # skip new points closer than this (removes jitter)
STROKE_SMOOTH_WINDOW = 5                 # points averaged together when a finished stroke is tidied up
PROGRESS_BAR_COLOUR = (235, 235, 235)   # light grey progress bar
PROGRESS_BAR_HEIGHT = 3                 # in design points (scaled by ui.px)

# --- Camera preview (top-right corner) ----------------------------------------
PREVIEW_SIZE = (288, 162)               # preview size in design points
HUD_MARGIN = 16                         # gap between HUD panels and the screen edge
TOAST_TOP = 36                          # distance of the toast from the top edge
SKELETON_COLOUR = (255, 255, 255)       # white lines between landmarks
JOINT_COLOUR = (0, 140, 255)            # orange landmark dots
SKELETON_WIDTH = 2
JOINT_RADIUS = 4
PREVIEW_BORDER_COLOUR = (255, 255, 255)
PREVIEW_BORDER_WIDTH = 2

# --- Keyboard codes from cv2.waitKey ------------------------------------------
KEY_ESC = 27
QUIT_KEYS = (ord("q"), KEY_ESC)
NEXT_KEYS = (3, 83, ord("d"))           # right arrow (code 3 on macOS, 83 on Linux) or d
PREV_KEYS = (2, 81, ord("a"))           # left arrow (code 2 on macOS, 81 on Linux) or a

# Text shown in the status pill and toast for each gesture.
LABEL = {Gesture.NEXT: "Next slide", Gesture.PREV: "Previous slide", Gesture.POINTER: "Pointer",
         Gesture.DRAW: "Drawing", Gesture.UNDO: "Undo", Gesture.PALM: "Open hand",
         Gesture.CLEAR: "Drawings cleared", Gesture.FIST: "Resting", Gesture.NONE: "Unknown sign"}

# Pairs of landmark indices joined by a line to draw the hand skeleton.
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Presenter:
    """Holds the slides, the current slide number and the drawings on each slide."""

    def __init__(self, slides: list[np.ndarray]):
        """Resize every slide to the W x H canvas once, up front."""
        self.slides = []
        for slide in slides:
            self.slides.append(deck.fit(slide, W, H))
        self.i = 0                    # index of the slide on screen
        # Drawings per slide: slide index -> list of strokes, each a list of points.
        self.strokes: dict[int, list[list[tuple[int, int]]]] = {}
        self.drawing = False          # True while a stroke is in progress

    @property
    def cur_strokes(self) -> list[list[tuple[int, int]]]:
        """Strokes on the current slide (an empty list is created if needed)."""
        return self.strokes.setdefault(self.i, [])

    def go(self, d: int) -> None:
        """Move d slides forward (or back if negative), staying within the deck."""
        last = len(self.slides) - 1
        self.i = max(0, min(last, self.i + d))
        self.end_stroke()

    def act(self, g: Gesture | None) -> None:
        """Apply a fired gesture action to the slides. None does nothing."""
        if g == Gesture.NEXT:
            self.go(1)
        elif g == Gesture.PREV:
            self.go(-1)
        elif g == Gesture.UNDO:
            if self.cur_strokes:
                self.cur_strokes.pop()
        elif g == Gesture.CLEAR:
            self.cur_strokes.clear()

    def draw_point(self, p: tuple[int, int]) -> None:
        """Add a point to the current stroke, starting a new stroke if needed."""
        if not self.drawing or not self.cur_strokes:
            self.cur_strokes.append([])
            self.drawing = True
        stroke = self.cur_strokes[-1]
        if not stroke:
            stroke.append(p)
            return
        # Skip points that barely moved, so a resting finger does not pile up
        # hundreds of identical points.
        last = stroke[-1]
        moved = abs(p[0] - last[0]) + abs(p[1] - last[1])
        if moved >= MIN_POINT_GAP_PX:
            stroke.append(p)

    def end_stroke(self) -> None:
        """Lift the pen. The finished stroke is smoothed so hand shake is less visible."""
        if self.drawing and self.cur_strokes:
            self.cur_strokes[-1] = smooth_stroke(self.cur_strokes[-1])
        self.drawing = False

    def render(self, pointer: tuple[int, int] | None = None) -> np.ndarray:
        """Return a new image: current slide + its drawings + pointer + progress bar."""
        frame = self.slides[self.i].copy()
        for stroke in self.cur_strokes:
            if len(stroke) > 1:
                points = np.array(stroke, np.int32)
                cv2.polylines(frame, [points], False, STROKE_COLOUR, ui.px(STROKE_WIDTH), cv2.LINE_AA)
        if pointer is not None:
            ui.pointer(frame, pointer, self.drawing)
        if len(self.slides) > 1:
            self._draw_progress_bar(frame)
        return frame

    def _draw_progress_bar(self, frame: np.ndarray) -> None:
        """Thin bar along the bottom edge showing how far through the deck we are."""
        bar_width = int(W * (self.i + 1) / len(self.slides))
        top = H - ui.px(PROGRESS_BAR_HEIGHT)
        cv2.rectangle(frame, (0, top), (bar_width, H), PROGRESS_BAR_COLOUR, -1)


def smooth_stroke(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Tidy a finished stroke with a moving average (each point becomes the mean of its neighbours).

    The first and last points are kept exactly, so the stroke still starts and
    ends where the user put it.
    """
    if len(points) < STROKE_SMOOTH_WINDOW:
        return points
    half = STROKE_SMOOTH_WINDOW // 2
    smoothed = [points[0]]
    for i in range(1, len(points) - 1):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        window = points[start:end]
        mean_x = sum(p[0] for p in window) / len(window)
        mean_y = sum(p[1] for p in window) / len(window)
        smoothed.append((int(round(mean_x)), int(round(mean_y))))
    smoothed.append(points[-1])
    return smoothed


def fit_to_window(frame: np.ndarray, slide: np.ndarray) -> np.ndarray:
    """Pad the 16:9 frame so it has the same shape as the window.

    A MacBook screen is 16:10, so in fullscreen there is spare space above and
    below the slide. Without this, the window shows that space as one black
    band. Here the slide is centred and the spare space is filled with the
    slide's own edge colours, so no band is visible.
    """
    try:
        _, _, window_w, window_h = cv2.getWindowImageRect(WIN)
    except cv2.error:
        return frame
    if window_w <= 0 or window_h <= 0:
        return frame

    wanted_h = int(round(W * window_h / window_w))    # frame height that matches the window shape
    wanted_w = int(round(H * window_w / window_h))    # or frame width, if the window is wider
    if wanted_h > H + 2:
        extra = wanted_h - H
        top = extra // 2
        bottom = extra - top
        top_colour = slide[0].mean(axis=0)
        bottom_colour = slide[-1].mean(axis=0)
        frame = cv2.copyMakeBorder(frame, top, 0, 0, 0, cv2.BORDER_CONSTANT, value=top_colour.tolist())
        frame = cv2.copyMakeBorder(frame, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=bottom_colour.tolist())
    elif wanted_w > W + 2:
        extra = wanted_w - W
        left = extra // 2
        right = extra - left
        left_colour = slide[:, 0].mean(axis=0)
        right_colour = slide[:, -1].mean(axis=0)
        frame = cv2.copyMakeBorder(frame, 0, 0, left, 0, cv2.BORDER_CONSTANT, value=left_colour.tolist())
        frame = cv2.copyMakeBorder(frame, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=right_colour.tolist())
    return frame


def _draw_camera_preview(frame: np.ndarray, cam: np.ndarray, lm) -> None:
    """Small camera image with the hand skeleton, in the top-right corner."""
    margin = ui.px(HUD_MARGIN)
    preview_w = ui.px(PREVIEW_SIZE[0])
    preview_h = ui.px(PREVIEW_SIZE[1])
    small = cv2.resize(cam, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

    if lm:
        # Landmarks are 0..1, so multiply by the preview size to get pixels.
        points = []
        for x, y in lm:
            points.append((int(x * preview_w), int(y * preview_h)))
        for a, b in HAND_LINKS:
            cv2.line(small, points[a], points[b], SKELETON_COLOUR, SKELETON_WIDTH, cv2.LINE_AA)
        for point in points:
            cv2.circle(small, point, JOINT_RADIUS, JOINT_COLOUR, -1, cv2.LINE_AA)

    left = W - preview_w - margin
    right = W - margin
    frame[margin:margin + preview_h, left:right] = small
    cv2.rectangle(frame, (left, margin), (right, margin + preview_h),
                  PREVIEW_BORDER_COLOUR, PREVIEW_BORDER_WIDTH, cv2.LINE_AA)


def overlay(frame, cam, lm, gesture, fingers, flash, idx, n, show_cam, show_help):
    """Draw the camera preview, status pill, help card and action toast."""
    margin = ui.px(HUD_MARGIN)
    if show_cam:
        _draw_camera_preview(frame, cam, lm)

    # Status pill, bottom-left.
    if lm:
        status = LABEL[gesture]
    else:
        status = "Show your hand to the camera"
    page = f"{idx + 1} / {n}"
    # fingers is turned into a tuple because the cached status_pill needs
    # hashable arguments.
    pill = ui.status_pill(status, page, tuple(fingers), gesture in CONTINUOUS)
    pill_top = H - margin - pill.shape[0]
    ui.blit(frame, pill, margin, pill_top)

    # Help card, stacked just above the pill.
    if show_help:
        card = ui.help_card()
        ui.blit(frame, card, margin, pill_top - margin - card.shape[0])

    # Toast, centred near the top.
    if flash:
        message = ui.toast(flash)
        ui.blit(frame, message, (W - message.shape[1]) // 2, ui.px(TOAST_TOP))


class Camera:
    """Reads the webcam on a background thread and always hands out the newest frame.

    Without this, frames queue up inside the driver while a slide is being drawn and
    the hand on screen lags behind the real hand.
    """

    def __init__(self, cap):
        """Start reading from an opened cv2.VideoCapture in a daemon thread."""
        self.cap = cap
        self.frame = None             # newest frame read
        self.seq = 0                  # counts frames, so callers can tell new from old
        self.alive = True             # set to False to stop the thread
        self.lock = threading.Lock()  # guards frame and seq between the two threads
        # daemon=True: the thread will not keep the program alive on exit.
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        """Thread body: keep reading frames until close() is called."""
        while self.alive:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(CAMERA_THREAD_IDLE_S)
                continue
            with self.lock:
                self.frame = frame
                self.seq = self.seq + 1

    def latest(self, last_seq: int, timeout: float = FRAME_TIMEOUT_S):
        """Wait for a frame newer than last_seq. Returns (frame, seq) or (None, last_seq)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                if self.seq != last_seq and self.frame is not None:
                    return self.frame, self.seq
            time.sleep(FRAME_POLL_S)
        return None, last_seq

    def close(self) -> None:
        """Stop the reader thread and release the webcam."""
        self.alive = False
        time.sleep(CAMERA_STOP_WAIT_S)
        self.cap.release()


def open_camera(index: int, wait_s: float = CAMERA_PERMISSION_WAIT_S):
    """Open the camera, waiting while macOS shows the first-run permission prompt."""
    deadline = time.monotonic() + wait_s
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened() and cap.read()[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            # A 1-frame buffer stops the driver holding on to old frames.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        if time.monotonic() > deadline:
            return None
        time.sleep(CAMERA_RETRY_S)


def fail(msg: str):
    """Exit with a message; show a dialog when running as a double-clicked app."""
    # A double-clicked .app has no terminal, so a printed error would be invisible.
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        safe = msg.replace('"', "'")
        subprocess.run(["osascript", "-e", f'display alert "Gesture Presenter" message "{safe}" as critical'])
    raise SystemExit(msg)


class ViewState:
    """Screen options the user can toggle, plus the current toast message."""

    def __init__(self, now: float):
        """Start with camera preview and help card shown, windowed mode."""
        self.show_cam = True
        self.show_help = True
        self.fullscreen = False
        self.help_until: float | None = now + HELP_AUTO_HIDE_S   # None = never auto-hide
        self.flash = ""                 # toast text
        self.flash_until = 0.0          # toast disappears after this time

    def auto_hide_help(self, now: float) -> None:
        """Hide the help card once its display time is over."""
        if self.help_until and now > self.help_until:
            self.show_help = False
            self.help_until = None

    def show_toast(self, text: str, now: float) -> None:
        """Show `text` in the toast for TOAST_DURATION_S seconds."""
        self.flash = text
        self.flash_until = now + TOAST_DURATION_S

    def current_toast(self, now: float) -> str:
        """Toast text to draw now, or "" if it has expired."""
        if now < self.flash_until:
            return self.flash
        return ""


def parse_args(argv=None) -> argparse.Namespace:
    """Read the presentation path and camera number from the command line."""
    parser = argparse.ArgumentParser(description="Hand Gesture Presentation System")
    parser.add_argument("file", nargs="?", help="presentation: " + " ".join(deck.SUPPORTED))
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args(argv)


def open_presentation(path: str) -> Presenter:
    """Load the file into a Presenter, or exit with a readable error."""
    try:
        return Presenter(deck.load(path))
    except Exception as e:
        fail(f"Could not open {path}:\n{e}")


def create_window() -> None:
    """Open a resizable window for the slides."""
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WINDOW_START_SIZE[0], WINDOW_START_SIZE[1])


def find_hand(tracker: HandTracker, cam: np.ndarray):
    """Run the hand tracker on a smaller RGB copy of the camera image."""
    # A smaller image makes MediaPipe faster; landmarks are 0..1 so the size
    # does not change their meaning.
    small = cv2.resize(cam, TRACKING_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   # OpenCV gives BGR, MediaPipe wants RGB
    return tracker.process(rgb)


def update_pointer(pres: Presenter, smooth: Smoother, pose: Gesture, lm):
    """Move the cursor and extend the drawing. Returns the cursor position or None.

    The index fingertip drives the cursor in both poses. Pointer (two fingers)
    moves the cursor with the pen up; Draw (index only) puts the pen down at
    that same spot. This is what lets the user write separate letters.
    """
    if pose not in CONTINUOUS or not lm:
        # No pointer pose: end any stroke and let the cursor jump next time.
        pres.end_stroke()
        smooth.reset()
        return None

    tip = lm[INDEX_TIP]
    pointer = smooth.map(tip[0], tip[1], W, H)
    if pose == Gesture.DRAW:
        pres.draw_point(pointer)
    else:
        pres.end_stroke()
    return pointer


def toggle_fullscreen(view: ViewState) -> None:
    """Switch the window between fullscreen and normal."""
    view.fullscreen = not view.fullscreen
    if view.fullscreen:
        mode = cv2.WINDOW_FULLSCREEN
    else:
        mode = cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, mode)


def handle_key(key: int, pres: Presenter, view: ViewState) -> bool:
    """React to one key press. Returns False when the user wants to quit."""
    if key in QUIT_KEYS:
        return False
    if key == ord("c"):
        view.show_cam = not view.show_cam
    elif key == ord("h"):
        view.show_help = not view.show_help
        view.help_until = None          # the user chose; stop auto-hiding
    elif key == ord("f"):
        toggle_fullscreen(view)
    elif key in NEXT_KEYS:
        pres.go(1)
    elif key in PREV_KEYS:
        pres.go(-1)
    return True


def run_loop(camera: Camera, tracker: HandTracker, engine: GestureEngine,
             smooth: Smoother, pres: Presenter, view: ViewState) -> None:
    """The per-frame pipeline. Runs until the user quits or the camera stops."""
    seq = 0
    while True:
        # 1. Newest camera frame, mirrored so moving right looks like right.
        cam, seq = camera.latest(seq)
        if cam is None:
            break
        cam = cv2.flip(cam, 1)

        # 2. Hand landmarks (None if no hand).
        lm = find_hand(tracker, cam)

        # 3. Steady pose and, sometimes, a fired action.
        now = time.monotonic()
        pose, fired = engine.update(lm, now)
        view.auto_hide_help(now)

        # 4. Apply the action to the slides.
        pres.act(fired)
        if fired:
            view.show_toast(LABEL[fired], now)

        # 5. Pointer and drawing.
        pointer = update_pointer(pres, smooth, pose, lm)

        # 6. Draw everything.
        out = pres.render(pointer)
        overlay(out, cam, lm, pose, engine.fingers, view.current_toast(now),
                pres.i, len(pres.slides), view.show_cam, view.show_help)

        # 7. Show it and handle the keyboard.
        cv2.imshow(WIN, fit_to_window(out, pres.slides[pres.i]))
        key = cv2.waitKey(1) & 0xFF
        if not handle_key(key, pres, view):
            break


def main(argv=None):
    """Program entry point: pick a file, open the camera, run the loop, clean up."""
    args = parse_args(argv)
    path = args.file or deck.pick_file()
    if not path:
        return
    pres = open_presentation(path)

    cap = open_camera(args.camera)
    if cap is None:
        fail("Cannot open the camera. Allow it in System Settings > Privacy & Security > Camera, then reopen the app.")

    camera = Camera(cap)
    tracker = HandTracker()
    engine = GestureEngine()
    smooth = Smoother()
    create_window()
    view = ViewState(time.monotonic())
    try:
        run_loop(camera, tracker, engine, smooth, pres, view)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
