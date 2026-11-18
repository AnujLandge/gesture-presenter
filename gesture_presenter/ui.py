"""On-screen graphics (HUD): step 6 of the pipeline, drawn on top of the slide.

Provides the status pill, help card, toasts and the laser pointer dot.

Text is drawn with Pillow (anti-aliased system fonts) into small RGBA images
that are cached and alpha-blended onto the slide, so the HUD costs almost
nothing per frame. OpenCV's own text drawing looks jagged, which is why Pillow
is used for text.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Scale and colours (RGBA for Pillow panels) ------------------------------
S = 1.5                                   # UI scale (canvas is 1920x1080)
INK = (245, 246, 248, 255)                # main text colour
MUTED = (160, 166, 178, 255)              # secondary text colour
ACCENT = (255, 255, 255, 255)             # highlighted text / raised finger dot
PANEL = (18, 20, 26, 218)                 # dark, slightly see-through panel fill
PANEL_OUTLINE = (255, 255, 255, 40)       # faint panel border and divider line
FINGER_DOWN = (70, 74, 84, 255)           # dot colour for a lowered finger
FINGER_LETTER = (10, 12, 16, 255)         # letter drawn inside each finger dot

# --- Pointer colours (BGR for OpenCV) -----------------------------------------
POINTER_GLOW = (40, 40, 255)              # soft red halo around the dot
POINTER_DOT = (30, 30, 235)               # solid red dot
POINTER_RING = (255, 255, 255)            # thin white ring around the dot
GLOW_OPACITY = 0.28                       # how strongly the halo shows through
GLOW_SIZE = 2.2                           # halo radius as a multiple of the dot radius

# --- Font files, tried in order (macOS, Windows, Linux) -----------------------
_REGULAR = ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_BOLD = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
         "C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# --- Help card text -----------------------------------------------------------
GESTURES = [("Open hand, swipe left", "Next slide"), ("Open hand, swipe right", "Previous slide"),
            ("Index + middle fingers", "Laser pointer"), ("Index finger", "Draw"),
            ("Three fingers", "Undo last drawing"), ("Open hand, hold still", "Clear drawings"),
            ("Fist", "Rest (nothing happens)")]
KEYS = "F  fullscreen      C  camera      H  show / hide this      Q  quit"
FINGER_LETTERS = "TIMRP"                  # thumb, index, middle, ring, pinky


def px(v: float) -> int:
    """Scale a design size (in points) to canvas pixels."""
    return int(round(v * S))


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False):
    """Load the first available system font at `size` points (cached)."""
    candidates = _BOLD if bold else _REGULAR
    for path in candidates:
        # In Helvetica.ttc the bold face is the second font (index 1).
        if bold and path.endswith(".ttc"):
            face_index = 1
        else:
            face_index = 0
        try:
            return ImageFont.truetype(path, px(size), index=face_index)
        except OSError:
            continue
    return ImageFont.load_default(px(size))


def _panel(w: int, h: int, radius: int = 14) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Make a transparent w x h image with a rounded dark panel on it."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), px(radius), fill=PANEL, outline=PANEL_OUTLINE)
    return img, draw


def _bgra(img: Image.Image) -> np.ndarray:
    """Convert a Pillow RGBA image to an OpenCV BGRA array."""
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)


# Cached because the same few combinations repeat frame after frame.
@lru_cache(maxsize=256)
def status_pill(status: str, page: str, fingers: tuple, active: bool) -> np.ndarray:
    """Bottom bar: current gesture, page number and the five finger dots."""
    regular = font(15)
    bold = font(15, True)
    status_w = int(bold.getlength(status))
    page_w = int(regular.getlength(page))
    gap = px(24)
    dot_step = px(24)
    w = px(28) + status_w + gap + page_w + gap + px(5 * 24) + px(10)
    h = px(44)
    img, draw = _panel(w, h, 22)

    x = px(18)
    cy = h // 2
    if active:
        status_colour = ACCENT
    else:
        status_colour = INK
    draw.text((x, cy), status, font=bold, fill=status_colour, anchor="lm")
    x += status_w + gap
    draw.text((x, cy), page, font=regular, fill=MUTED, anchor="lm")
    x += page_w + gap

    # One dot per finger, lit when the app sees that finger raised. This lets
    # the presenter see exactly what the classifier sees.
    radius = px(9)
    for letter, up in zip(FINGER_LETTERS, fingers):
        if up:
            dot_colour = ACCENT
        else:
            dot_colour = FINGER_DOWN
        draw.ellipse((x, cy - radius, x + 2 * radius, cy + radius), fill=dot_colour)
        draw.text((x + radius, cy), letter, font=font(9, True), fill=FINGER_LETTER, anchor="mm")
        x += dot_step
    return _bgra(img)


@lru_cache(maxsize=1)
def help_card() -> np.ndarray:
    """Panel listing every gesture and keyboard shortcut (drawn once, cached)."""
    row = px(30)
    pad = px(20)
    w = px(470)
    h = pad * 2 + px(34) + row * len(GESTURES) + px(34)
    img, draw = _panel(w, h)

    draw.text((pad, pad), "Hand gestures", font=font(17, True), fill=INK)
    y = pad + px(38)
    for sign, action in GESTURES:
        draw.text((pad, y), sign, font=font(14), fill=MUTED)
        draw.text((w - pad, y), action, font=font(14, True), fill=INK, anchor="ra")
        y += row
    draw.line((pad, y + px(4), w - pad, y + px(4)), fill=PANEL_OUTLINE)
    draw.text((pad, y + px(14)), KEYS, font=font(11), fill=MUTED)
    return _bgra(img)


@lru_cache(maxsize=32)
def toast(text: str) -> np.ndarray:
    """Short pop-up message such as "Next slide" (cached per text)."""
    f = font(24, True)
    w = int(f.getlength(text)) + px(56)
    h = px(60)
    img, draw = _panel(w, h, 30)
    draw.text((w // 2, h // 2), text, font=f, fill=ACCENT, anchor="mm")
    return _bgra(img)


def blit(frame: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA image onto a BGR frame (clipped to the frame)."""
    h, w = img.shape[:2]
    # Keep the image fully inside the frame.
    x = max(0, min(x, frame.shape[1] - w))
    y = max(0, min(y, frame.shape[0] - h))
    region = frame[y:y + h, x:x + w]
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    # result = image * alpha + background * (1 - alpha), written in place.
    region[:] = (img[:, :, :3] * alpha + region * (1.0 - alpha)).astype(np.uint8)


def pointer(frame: np.ndarray, p: tuple[int, int], drawing: bool) -> None:
    """Laser dot with a soft glow (smaller, solid dot while drawing)."""
    if drawing:
        radius = px(4)
    else:
        radius = px(6)
    glow = int(radius * GLOW_SIZE)

    # Blend the glow only in a small square around the dot, which is much
    # cheaper than blending the whole 1920x1080 frame.
    x0 = max(p[0] - glow, 0)
    y0 = max(p[1] - glow, 0)
    region = frame[y0:p[1] + glow, x0:p[0] + glow]
    if region.size:
        glow_layer = region.copy()
        cv2.circle(glow_layer, (p[0] - x0, p[1] - y0), glow - 2, POINTER_GLOW, -1, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, GLOW_OPACITY, region, 1 - GLOW_OPACITY, 0, region)

    cv2.circle(frame, p, radius, POINTER_DOT, -1, cv2.LINE_AA)
    cv2.circle(frame, p, radius, POINTER_RING, max(1, px(1.5)), cv2.LINE_AA)
