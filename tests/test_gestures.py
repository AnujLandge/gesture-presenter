"""Unit tests for gesture_presenter.gestures.

No camera is needed: `hand()` builds a fake set of 21 landmarks with chosen
fingers raised, so every rule can be checked with exact, repeatable input.
"""
import math

from gesture_presenter.gestures import Config, Gesture, GestureEngine, Smoother, classify, fingers_up

FPS = 30                      # simulated camera frame rate
FRAME_TIME = 1 / FPS          # seconds between simulated frames

# Finger patterns used by several tests: [thumb, index, middle, ring, pinky].
THUMB_ONLY = [1, 0, 0, 0, 0]
PINKY_ONLY = [0, 0, 0, 0, 1]
INDEX_ONLY = [0, 1, 0, 0, 0]
TWO_FINGERS = [0, 1, 1, 0, 0]
OPEN_HAND = [1, 1, 1, 1, 1]
FIST = [0, 0, 0, 0, 0]

POSES = (THUMB_ONLY, TWO_FINGERS, OPEN_HAND, FIST, PINKY_ONLY, INDEX_ONLY)

# Positions of the fixed palm points relative to the wrist (y grows downwards).
PALM_POINTS = {0: (0, 0), 5: (-0.05, -0.10), 9: (0, -0.11), 13: (0.04, -0.10), 17: (0.08, -0.09),
               3: (-0.09, -0.06)}
THUMB_TIP_UP = (-0.15, -0.10)
THUMB_TIP_DOWN = (-0.03, -0.07)
# (tip index, PIP index, x position) for index, middle, ring and pinky fingers.
FINGERS = ((8, 6, -0.05), (12, 10, 0.0), (16, 14, 0.04), (20, 18, 0.08))


def hand(up, cx=0.5, wrist_y=0.6, angle=0.0):
    """Synthetic hand. up = [thumb, index, middle, ring, pinky]; angle rotates it (radians)."""
    pts = dict(PALM_POINTS)
    if up[0]:
        pts[4] = THUMB_TIP_UP
    else:
        pts[4] = THUMB_TIP_DOWN
    for finger_number, (tip, pip, x) in enumerate(FINGERS, start=1):
        pts[pip] = (x, -0.16)
        if up[finger_number]:
            pts[tip] = (x, -0.25)     # straight: tip beyond the middle joint
        else:
            pts[tip] = (x, -0.09)     # curled: tip folded back towards the palm

    # Rotate every point around the wrist, then move the wrist to (cx, wrist_y).
    lm = [(cx, wrist_y)] * 21
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for i, (x, y) in pts.items():
        lm[i] = (cx + x * cos_a - y * sin_a, wrist_y + x * sin_a + y * cos_a)
    return lm


def run(engine, lm_fn, frames, t0=0.0, dt=FRAME_TIME):
    """Feed `frames` frames of lm_fn(k) to the engine. Returns (fired actions, end time)."""
    fired = []
    for k in range(frames):
        _, action = engine.update(lm_fn(k), t0 + k * dt)
        if action:
            fired.append(action)
    return fired, t0 + frames * dt


def still(up, **kw):
    """Frame source for a hand that holds the same pose in the same place."""
    def frame(k):
        return hand(up, **kw)
    return frame


# --- Finger test and pose table -----------------------------------------------

def test_fingers_up_roundtrip():
    """fingers_up reads back exactly the fingers the synthetic hand raised."""
    for up in POSES:
        assert fingers_up(hand(up)) == up


def test_fingers_up_rotation_invariant():
    """Tilted and upside-down hands give the same finger reading."""
    for angle in (-1.2, -0.6, 0.6, 1.2, 3.14):
        for up in POSES:
            assert fingers_up(hand(up, angle=angle)) == up, (angle, up)


def test_classify():
    """Each finger pattern maps to the right pose; unknown patterns give NONE."""
    assert classify(hand(PINKY_ONLY)) == Gesture.NEXT
    assert classify(hand(THUMB_ONLY)) == Gesture.PREV
    assert classify(hand(INDEX_ONLY)) == Gesture.DRAW
    assert classify(hand(TWO_FINGERS)) == Gesture.POINTER
    assert classify(hand([0, 1, 1, 1, 0])) == Gesture.UNDO
    assert classify(hand(OPEN_HAND)) == Gesture.PALM
    assert classify(hand(FIST)) == Gesture.FIST
    assert classify(hand([0, 0, 1, 0, 1])) == Gesture.NONE


def test_classify_ignores_thumb_for_multi_finger_poses():
    """The thumb is unreliable, so DRAW/POINTER/UNDO/PALM accept it up or down."""
    assert classify(hand([1, 1, 0, 0, 0])) == Gesture.DRAW
    assert classify(hand([1, 1, 1, 0, 0])) == Gesture.POINTER
    assert classify(hand([1, 1, 1, 1, 0])) == Gesture.UNDO
    assert classify(hand([0, 1, 1, 1, 1])) == Gesture.PALM


# --- Debounce, cooldown and re-arming -----------------------------------------

def test_pose_fires_once_while_held():
    """Holding a discrete pose for 3 s gives exactly one action."""
    e = GestureEngine()
    fired, _ = run(e, still(PINKY_ONLY), 90)
    assert fired == [Gesture.NEXT]


def test_pose_needs_stable_frames():
    """A discrete pose shown for fewer than stable_frames frames does not fire."""
    e = GestureEngine()
    fired, _ = run(e, still(PINKY_ONLY), Config().stable_frames - 1)
    assert fired == []


def test_pose_works_low_in_frame():
    """A hand near the bottom edge of the image is still recognised."""
    e = GestureEngine()
    fired, _ = run(e, still(PINKY_ONLY, wrist_y=0.95), 30)
    assert fired == [Gesture.NEXT]


def test_rearm_after_release_and_cooldown():
    """After a fist (release) and the cooldown, the same pose fires again."""
    e = GestureEngine(Config(cooldown_s=0.5))
    f1, t = run(e, still(PINKY_ONLY), 15)
    _, t = run(e, still(FIST), 20, t)
    f2, _ = run(e, still(PINKY_ONLY), 15, t)
    assert f1 == [Gesture.NEXT]
    assert f2 == [Gesture.NEXT]


def test_cooldown_blocks_quick_second_action():
    """Even after releasing, a second action inside the cooldown is ignored."""
    e = GestureEngine()                       # default cooldown 1.2 s
    f1, t = run(e, still(PINKY_ONLY), 10)
    _, t = run(e, still(FIST), 3, t)          # quick release
    f2, _ = run(e, still(PINKY_ONLY), 10, t)  # still inside 1.2 s of the first action
    assert f1 == [Gesture.NEXT]
    assert f2 == []


# --- Swipes and clear ---------------------------------------------------------

def test_swipe_left_is_next_and_return_stroke_ignored():
    """A fast open-hand move left is NEXT; the hand swinging back does nothing."""
    e = GestureEngine()
    out, t = run(e, lambda k: hand(OPEN_HAND, cx=0.75 - 0.04 * k), 12)
    back, _ = run(e, lambda k: hand(OPEN_HAND, cx=0.30 + 0.04 * k), 12, t)
    assert out == [Gesture.NEXT]
    assert back == []


def test_swipe_right_is_prev():
    """An open-hand move to the right is PREV."""
    e = GestureEngine()
    fired, _ = run(e, lambda k: hand([0, 1, 1, 1, 1], cx=0.25 + 0.04 * k), 12)
    assert fired == [Gesture.PREV]


def test_slow_drift_is_not_a_swipe():
    """Moving slowly (less than swipe_dist within swipe_window_s) never swipes."""
    e = GestureEngine(Config(clear_hold_s=99))
    fired, _ = run(e, lambda k: hand(OPEN_HAND, cx=0.3 + 0.003 * k), 100)
    assert fired == []


def test_vertical_move_is_not_a_swipe():
    """An up/down movement of the open hand is not treated as a swipe."""
    e = GestureEngine(Config(clear_hold_s=99))
    fired, _ = run(e, lambda k: hand(OPEN_HAND, wrist_y=0.95 - 0.04 * k), 14)
    assert fired == []


def test_slow_deliberate_swipe_counts():
    """A steady swipe lasting about 0.8 s still counts (window is 0.9 s)."""
    e = GestureEngine()
    fired, _ = run(e, lambda k: hand(OPEN_HAND, cx=0.75 - 0.016 * k), 30)
    assert fired == [Gesture.NEXT]


def test_clear_requires_still_hold():
    """Holding the open hand still clears drawings, but only after clear_hold_s."""
    e = GestureEngine(Config(clear_hold_s=1.0))
    short, _ = run(e, still(OPEN_HAND), 20)        # 0.66 s: too short
    assert short == []
    e = GestureEngine(Config(clear_hold_s=1.0))
    held, _ = run(e, still(OPEN_HAND), 90)         # 3 s: fires exactly once
    assert held == [Gesture.CLEAR]


# --- Robustness ---------------------------------------------------------------

def test_no_hand_resets():
    """With no hand the engine reports NONE, no action and all fingers down."""
    e = GestureEngine()
    assert e.update(None, 0.0) == (Gesture.NONE, None)
    assert e.fingers == [0, 0, 0, 0, 0]


def test_single_misread_frame_does_not_break_drawing():
    """One garbage frame in the middle of a stroke is outvoted by its neighbours."""
    e = GestureEngine()
    poses = []
    for k in range(30):
        if k == 15:
            up = [0, 0, 1, 0, 1]
        else:
            up = INDEX_ONLY
        pose, _ = e.update(hand(up), k * FRAME_TIME)
        poses.append(pose)
    for pose in poses:
        assert pose == Gesture.DRAW


def test_majority_vote_tie_goes_to_newest_pose():
    """When two poses are tied in the vote, the more recent one wins."""
    e = GestureEngine()
    sequence = [INDEX_ONLY, INDEX_ONLY, TWO_FINGERS, TWO_FINGERS]   # 2 vs 2
    pose = None
    for k, up in enumerate(sequence):
        pose, _ = e.update(hand(up), k * FRAME_TIME)
    assert pose == Gesture.POINTER


# --- Cursor smoothing ---------------------------------------------------------

def test_smoother_bounds():
    """The camera margin maps to the screen edges, so corners are reachable."""
    s = Smoother(min_alpha=1.0, max_alpha=1.0, margin=0.15)
    assert s.map(0.0, 0.0, 1280, 720) == (0, 0)
    assert s.map(1.0, 1.0, 1280, 720) == (1280, 720)


def test_smoother_ignores_tremor_and_follows_fast_moves():
    """Tiny jitter is ignored; a large move is followed within a few frames."""
    s = Smoother()
    start = s.map(0.5, 0.5, 1920, 1080)
    assert s.map(0.5005, 0.5, 1920, 1080) == start     # inside the 3 px dead zone
    end = start
    for _ in range(12):
        end = s.map(0.7, 0.5, 1920, 1080)
    assert end[0] > start[0] + 500


def test_smoother_reset_jumps_to_hand():
    """After reset() the first position is used directly, with no smoothing."""
    s = Smoother()
    s.map(0.2, 0.2, 1920, 1080)
    s.reset()
    assert s.map(0.5, 0.5, 1920, 1080) == (960, 540)
