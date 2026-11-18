"""Gesture recognition: step 3 of the pipeline (landmarks -> poses -> actions).

The hand tracker gives us 21 landmarks per frame. This module turns them into:

1. which fingers are raised (``fingers_up``),
2. a named hand pose such as DRAW or POINTER (``classify``),
3. a steady pose and, now and then, a one-off action such as NEXT slide
   (``GestureEngine``), and
4. a smooth on-screen cursor position (``Smoother``).

This file is pure Python (no OpenCV or MediaPipe imports), so every rule here
can be unit tested with made-up hands, without a camera.

Landmarks are (x, y) tuples normalised to the range 0..1 in the mirrored
camera image. All finger tests compare distances instead of "is the tip above
the knuckle", so they work for the left or right hand and at any rotation.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

# --- MediaPipe landmark indices (0 = wrist, then 4 points per finger) --------
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# --- Finger test thresholds --------------------------------------------------
# A finger counts as raised when its tip is this many times further from the
# wrist than its middle joint (PIP) is, i.e. a 12% safety margin.
FINGER_EXTENSION_RATIO = 1.12
# The thumb is raised when its tip is this many times further from the pinky
# knuckle than the thumb's own IP joint is (15% margin)...
THUMB_EXTENSION_RATIO = 1.15
# ...and its tip is at least this fraction of the palm length away from the
# index knuckle (so a thumb tucked against the index finger is "down").
THUMB_MIN_GAP_PALM_FRACTION = 0.45
# Fallback palm length if the wrist and middle knuckle land on the same point,
# so the thumb threshold never becomes exactly zero.
MIN_PALM_LENGTH = 1e-6

# --- Swipe shape --------------------------------------------------------------
# A swipe must be mostly sideways: vertical travel has to stay below this
# fraction of the horizontal travel.
SWIPE_MAX_VERTICAL_RATIO = 0.6

# --- Cursor smoothing defaults (used by Smoother) -----------------------------
SMOOTH_MIN_ALPHA = 0.30   # follow strength when the hand moves slowly (steady)
SMOOTH_MAX_ALPHA = 0.8    # follow strength when the hand moves fast (no lag)
SMOOTH_MARGIN = 0.12      # camera border ignored on each side, so screen edges are reachable
SMOOTH_DEAD_PX = 3.0      # moves smaller than this (pixels) are treated as tremor
SMOOTH_FAST_PX = 90.0    # moves this big (pixels) or larger get the maximum alpha

# Value used for "no action has fired yet", far enough in the past that the
# first action is never blocked by the cooldown.
NEVER = -1e9


class Gesture(str, Enum):
    """Every pose the system can recognise, plus the CLEAR action."""

    NONE = "none"            # fingers do not match any known pose
    NEXT = "next"            # swipe left, or pinky only
    PREV = "prev"            # swipe right, or thumb only
    POINTER = "pointer"      # index + middle
    DRAW = "draw"            # index only
    UNDO = "undo"            # index + middle + ring
    PALM = "palm"            # open hand: swipe to navigate, hold still to clear
    CLEAR = "clear"          # fired action only (open hand held still)
    FIST = "fist"            # idle


# Poses that trigger a single action (fire once, then wait).
DISCRETE = {Gesture.NEXT, Gesture.PREV, Gesture.UNDO}
# Poses that act on every frame while they are held (cursor follows the hand).
CONTINUOUS = {Gesture.POINTER, Gesture.DRAW}

# Pose table: (thumb, index, middle, ring, pinky) -> pose.
# The thumb is often misread, so most poses accept it either up or down.
POSE_TABLE = {
    (1, 0, 0, 0, 0): Gesture.PREV,
    (0, 0, 0, 0, 1): Gesture.NEXT,
    (0, 1, 0, 0, 0): Gesture.DRAW,
    (1, 1, 0, 0, 0): Gesture.DRAW,
    (0, 1, 1, 0, 0): Gesture.POINTER,
    (1, 1, 1, 0, 0): Gesture.POINTER,
    (0, 1, 1, 1, 0): Gesture.UNDO,
    (1, 1, 1, 1, 0): Gesture.UNDO,
    (0, 1, 1, 1, 1): Gesture.PALM,
    (1, 1, 1, 1, 1): Gesture.PALM,
    (0, 0, 0, 0, 0): Gesture.FIST,
}

# (tip, PIP joint) pairs for the four fingers, in order index -> pinky.
FINGER_JOINTS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)


def _dist(a, b) -> float:
    """Straight-line distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _thumb_is_up(lm) -> int:
    """Return 1 if the thumb is stretched out, else 0."""
    palm_length = _dist(lm[WRIST], lm[MIDDLE_MCP]) or MIN_PALM_LENGTH
    tip_to_pinky = _dist(lm[THUMB_TIP], lm[PINKY_MCP])
    joint_to_pinky = _dist(lm[THUMB_IP], lm[PINKY_MCP])
    tip_to_index = _dist(lm[THUMB_TIP], lm[INDEX_MCP])

    # The thumb bends sideways, not towards the wrist, so we measure it against
    # the far side of the palm (pinky knuckle) instead of the wrist.
    sticks_out = tip_to_pinky > joint_to_pinky * THUMB_EXTENSION_RATIO
    away_from_index = tip_to_index > THUMB_MIN_GAP_PALM_FRACTION * palm_length
    return int(sticks_out and away_from_index)


def _finger_is_up(lm, tip: int, pip: int) -> int:
    """Return 1 if the finger with this tip and PIP joint is stretched out."""
    # A straight finger's tip is further from the wrist than its middle joint;
    # a curled finger's tip folds back towards the wrist. Comparing distances
    # (not y-coordinates) means the test still works if the hand is tilted.
    wrist = lm[WRIST]
    tip_distance = _dist(lm[tip], wrist)
    joint_distance = _dist(lm[pip], wrist)
    return int(tip_distance > joint_distance * FINGER_EXTENSION_RATIO)


def fingers_up(lm) -> list[int]:
    """Return [thumb, index, middle, ring, pinky] as 0/1 (rotation invariant)."""
    result = [_thumb_is_up(lm)]
    for tip, pip in FINGER_JOINTS:
        result.append(_finger_is_up(lm, tip, pip))
    return result


def classify(lm) -> Gesture:
    """Name the pose of a single frame by looking up its finger pattern."""
    return POSE_TABLE.get(tuple(fingers_up(lm)), Gesture.NONE)


@dataclass
class Config:
    """Timing and distance thresholds for GestureEngine."""

    stable_frames: int = 8        # frames a pose must persist before it fires
    pose_frames: int = 5          # majority vote window that steadies the reported pose
    cooldown_s: float = 1.2       # min gap between discrete actions
    swipe_dist: float = 0.30      # open-hand travel (fraction of frame width) for a swipe
    swipe_window_s: float = 0.9   # ...within this time, so slow deliberate swipes count
    clear_hold_s: float = 1.5     # open hand held still this long clears drawings
    still_dist: float = 0.06      # max drift that still counts as "held still"


class GestureEngine:
    """Turns raw per-frame hand landmarks into a steady pose and debounced actions.

    Call ``update`` once per camera frame. It returns the steady pose (used for
    the status bar and the pointer) and, only on the frame where it happens,
    a fired action such as NEXT, PREV, UNDO or CLEAR.
    """

    def __init__(self, cfg: Config | None = None):
        """Create an engine with the given thresholds (defaults if None)."""
        self.cfg = cfg or Config()
        # Raw pose of each of the last few frames (newest on the right).
        self.history: deque[Gesture] = deque(maxlen=self.cfg.stable_frames)
        # (time, x, y) of the open hand, used to detect swipes.
        self.track: deque[tuple[float, float, float]] = deque()
        # Latest finger states, shown as T I M R P in the status bar.
        self.fingers = [0, 0, 0, 0, 0]
        # Time of the last fired action, for the cooldown.
        self.last_fire = NEVER
        # A pose must be released before it can fire again.
        self.armed = True
        # (start time, palm position) while the open hand is held still.
        self.hold: tuple[float, tuple[float, float]] | None = None

    def update(self, lm, now: float) -> tuple[Gesture, Gesture | None]:
        """Feed one frame. Returns (steady_pose, fired_action_or_None)."""
        if lm is None:
            self._reset_when_hand_lost()
            return Gesture.NONE, None

        self.fingers = fingers_up(lm)
        raw_pose = POSE_TABLE.get(tuple(self.fingers), Gesture.NONE)
        self.history.append(raw_pose)
        pose = self._pose()
        cooled_down = now - self.last_fire >= self.cfg.cooldown_s

        if pose == Gesture.PALM:
            fired = self._update_open_hand(lm[MIDDLE_MCP], now, cooled_down)
        else:
            fired = self._update_finger_pose(pose, cooled_down)

        if fired:
            self._after_fire(now)
        return pose, fired

    def _reset_when_hand_lost(self) -> None:
        """Forget all short-term state when no hand is visible."""
        self.history.clear()
        self.track.clear()
        self.armed = True
        self.hold = None
        self.fingers = [0, 0, 0, 0, 0]

    def _pose(self) -> Gesture:
        """Majority vote over the last few frames, so one misread frame changes nothing.

        The pose seen most often wins. If two poses are tied, the one seen most
        recently wins, so the system still reacts quickly to a real change.
        """
        recent = list(self.history)[-self.cfg.pose_frames:]
        best_pose = recent[-1]
        best_count = recent.count(best_pose)
        # Walk from newest to oldest; only a strictly higher count replaces the
        # current best, which gives ties to the more recent pose.
        for candidate in reversed(recent):
            count = recent.count(candidate)
            if count > best_count:
                best_pose = candidate
                best_count = count
        return best_pose

    def _update_open_hand(self, palm: tuple[float, float], now: float,
                          cooled_down: bool) -> Gesture | None:
        """Handle the open hand: a swipe changes slide, holding still clears."""
        if not cooled_down:
            # Right after a swipe the hand swings back; ignore that return stroke.
            self.track.clear()
            self.hold = None
            return None

        swipe = self._detect_swipe(palm, now)
        if swipe is not None:
            return swipe
        if self.armed:
            return self._detect_still_hold(palm, now)
        return None

    def _detect_swipe(self, palm: tuple[float, float], now: float) -> Gesture | None:
        """Return NEXT/PREV if the palm moved far enough sideways, recently enough."""
        self.track.append((now, palm[0], palm[1]))
        # Keep only positions from the last swipe_window_s seconds.
        while now - self.track[0][0] > self.cfg.swipe_window_s:
            self.track.popleft()

        _, start_x, start_y = self.track[0]
        dx = palm[0] - start_x
        dy = palm[1] - start_y
        far_enough = abs(dx) >= self.cfg.swipe_dist
        mostly_sideways = abs(dy) < abs(dx) * SWIPE_MAX_VERTICAL_RATIO
        if not (far_enough and mostly_sideways):
            return None
        # The image is mirrored, so moving the hand to the left gives dx < 0.
        if dx < 0:
            return Gesture.NEXT
        return Gesture.PREV

    def _detect_still_hold(self, palm: tuple[float, float], now: float) -> Gesture | None:
        """Return CLEAR once the open hand has stayed in one spot long enough."""
        if self.hold is None or _dist(palm, self.hold[1]) > self.cfg.still_dist:
            # First frame, or the hand drifted too far: start timing again here.
            self.hold = (now, palm)
            return None
        held_for = now - self.hold[0]
        if held_for >= self.cfg.clear_hold_s:
            return Gesture.CLEAR
        return None

    def _update_finger_pose(self, pose: Gesture, cooled_down: bool) -> Gesture | None:
        """Handle every pose except the open hand. Fires DISCRETE poses once."""
        self.track.clear()
        self.hold = None
        if pose not in DISCRETE:
            # Showing any non-action pose (fist, pointer, ...) re-arms the engine.
            self.armed = True
            return None

        # Debounce: the pose must be the raw reading in every one of the last
        # stable_frames frames, not just win the majority vote.
        history_full = len(self.history) == self.cfg.stable_frames
        all_same = True
        for past_pose in self.history:
            if past_pose != pose:
                all_same = False
                break
        stable = history_full and all_same

        if stable and cooled_down and self.armed:
            return pose
        return None

    def _after_fire(self, now: float) -> None:
        """Start the cooldown and disarm, so one gesture gives one action."""
        self.last_fire = now
        self.armed = False
        self.hold = None
        self.track.clear()


class Smoother:
    """Adaptive cursor smoothing: heavy when the hand is slow (steady, precise),
    light when it moves fast (no lag). Small tremors inside the dead zone are ignored."""

    def __init__(self, min_alpha: float = SMOOTH_MIN_ALPHA, max_alpha: float = SMOOTH_MAX_ALPHA,
                 margin: float = SMOOTH_MARGIN, dead_px: float = SMOOTH_DEAD_PX,
                 fast_px: float = SMOOTH_FAST_PX):
        """Store the smoothing settings. See the SMOOTH_* constants for meanings."""
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.margin = margin
        self.dead_px = dead_px
        self.fast_px = fast_px
        self.p: tuple[float, float] | None = None   # current smoothed cursor position

    def _to_screen(self, value: float, size: int) -> float:
        """Map a 0..1 camera coordinate to 0..size, ignoring the outer margin."""
        usable = 1 - 2 * self.margin
        fraction = (value - self.margin) / usable
        fraction = min(max(fraction, 0.0), 1.0)
        return fraction * size

    def map(self, x: float, y: float, w: int, h: int) -> tuple[int, int]:
        """Turn a normalised hand position into a smoothed w x h screen pixel."""
        target = (self._to_screen(x, w), self._to_screen(y, h))

        if self.p is None:
            # First frame: jump straight to the hand.
            self.p = target
        else:
            distance = _dist(target, self.p)
            if distance > self.dead_px:
                # Exponential smoothing: move a fraction alpha of the way to
                # the target. Bigger moves get a bigger alpha (less lag).
                speed = min(distance / self.fast_px, 1.0)
                alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * speed
                new_x = self.p[0] + alpha * (target[0] - self.p[0])
                new_y = self.p[1] + alpha * (target[1] - self.p[1])
                self.p = (new_x, new_y)
        return int(self.p[0]), int(self.p[1])

    def reset(self) -> None:
        """Forget the cursor position, so the next map() jumps to the hand."""
        self.p = None
