"""Draws Gesture Presenter app-icon concepts with Pillow. Usage: python make_icon.py [A|B|C]"""
import math
import sys
from PIL import Image, ImageDraw

SS = 4
SIZE = 1024 * SS


def s(v):
    return int(v * SS)


def background(top, bottom):
    """Rounded-square icon base with a vertical gradient."""
    grad = Image.new("RGBA", (SIZE, SIZE))
    gd = ImageDraw.Draw(grad)
    for y in range(SIZE):
        t = y / SIZE
        gd.line((0, y, SIZE, y), fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((s(100), s(100), s(924), s(924)), s(185), fill=255)
    icon = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    icon.paste(grad, (0, 0), mask)
    return icon


def dot(d, x, y, r, fill):
    d.ellipse((s(x - r), s(y - r), s(x + r), s(y + r)), fill=fill)


def line(d, pts, width, fill):
    d.line([(s(x), s(y)) for x, y in pts], fill=fill, width=s(width), joint="curve")
    for x, y in (pts[0], pts[-1]):
        dot(d, x, y, width / 2, fill)


# 21 hand landmarks (same layout MediaPipe uses), drawn as a skeleton
HAND = {0: (500, 790), 1: (392, 742), 2: (318, 662), 3: (268, 580), 4: (226, 508),
        5: (410, 520), 6: (392, 404), 7: (382, 330), 8: (374, 262),
        9: (500, 500), 10: (502, 370), 11: (504, 288), 12: (506, 214),
        13: (584, 520), 14: (606, 404), 15: (618, 332), 16: (628, 268),
        17: (656, 570), 18: (700, 480), 19: (726, 424), 20: (750, 372)}
BONES = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17)]


def concept_a():
    """Hand-tracking skeleton: white bones, orange joints."""
    icon = background((34, 30, 74), (13, 14, 34))
    d = ImageDraw.Draw(icon)
    for bone in BONES:
        line(d, [HAND[i] for i in bone], 22, (255, 255, 255, 255))
    for i, (x, y) in HAND.items():
        dot(d, x, y, 30 if i in (4, 8, 12, 16, 20) else 24, (255, 140, 40, 255))
        dot(d, x, y, 11, (255, 255, 255, 255))
    return icon


def concept_b():
    """Slide with a play triangle and swipe arcs."""
    icon = background((255, 122, 89), (214, 60, 90))
    d = ImageDraw.Draw(icon)
    d.rounded_rectangle((s(230), s(300), s(700), s(640)), s(48), fill=(255, 255, 255, 255))
    d.polygon([(s(410), s(390)), (s(410), s(550)), (s(545), s(470))], fill=(214, 60, 90, 255))
    line(d, [(465, 640), (465, 730)], 30, (255, 255, 255, 255))
    line(d, [(360, 740), (570, 740)], 30, (255, 255, 255, 255))
    for radius in (120, 190):
        box = (s(700 - radius), s(470 - radius), s(700 + radius), s(470 + radius))
        d.arc(box, -42, 42, fill=(255, 255, 255, 255), width=s(28))
    return icon


def concept_c():
    """Fingertip pointer: a dot with ripple rings and a sweep trail."""
    icon = background((28, 110, 140), (17, 34, 80))
    d = ImageDraw.Draw(icon)
    trail = [(250 + t * 330, 700 - 330 * math.sin(t * math.pi / 2)) for t in [i / 30 for i in range(31)]]
    for i in range(len(trail) - 1):
        line(d, trail[i:i + 2], 10 + 34 * i / 30, (255, 255, 255, int(60 + 195 * i / 30)))
    cx, cy = trail[-1]
    for radius, alpha in ((150, 70), (105, 130)):
        d.ellipse((s(cx - radius), s(cy - radius), s(cx + radius), s(cy + radius)), outline=(255, 255, 255, alpha), width=s(14))
    dot(d, cx, cy, 62, (255, 96, 86, 255))
    dot(d, cx, cy, 62 - 14, (255, 130, 110, 255))
    return icon


def small_hand(icon, cx, cy, scale, fill):
    """A simple open hand made of rounded bars (four fingers, a thumb and a palm)."""
    d = ImageDraw.Draw(icon)
    finger_w = 30 * scale
    for i, length in enumerate((120, 150, 140, 105)):
        x = cx - 68 * scale + i * 36 * scale
        top = cy - length * scale
        d.rounded_rectangle((s(x), s(top), s(x + finger_w), s(cy + 20 * scale)), s(finger_w / 2), fill=fill)
    d.rounded_rectangle((s(cx - 68 * scale), s(cy - 10 * scale), s(cx + 70 * scale), s(cy + 120 * scale)),
                        s(46 * scale), fill=fill)
    layer = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    tx, ty = cx - 52 * scale, cy + 86 * scale
    ld.rounded_rectangle((s(tx - 16 * scale), s(ty - 120 * scale), s(tx + 16 * scale), s(ty)), s(16 * scale), fill=fill)
    icon.alpha_composite(layer.rotate(48, center=(s(tx), s(ty)), resample=Image.BICUBIC))


def concept_d():
    """Presentation-app style: a slide on a stand with a chart, plus a hand badge."""
    icon = background((255, 138, 76), (226, 68, 62))
    d = ImageDraw.Draw(icon)
    white = (255, 255, 255, 255)
    line(d, [(512, 610), (512, 720)], 30, white)                    # stand
    line(d, [(400, 735), (624, 735)], 30, white)
    d.rounded_rectangle((s(215), s(245), s(809), s(625)), s(44), fill=white)   # slide
    bar = (226, 68, 62, 255)
    for x, h in ((300, 110), (390, 190), (480, 150)):               # bar chart on the slide
        d.rounded_rectangle((s(x), s(545 - h), s(x + 58), s(545)), s(12), fill=bar)
    for y, w in ((340, 150), (385, 110)):                           # text lines
        d.rounded_rectangle((s(590), s(y), s(590 + w), s(y + 24)), s(12), fill=(255, 190, 170, 255))
    dot(d, 735, 690, 132, (255, 255, 255, 255))                     # hand badge
    dot(d, 735, 690, 112, (44, 40, 96, 255))
    small_hand(icon, 738, 690, 0.62, white)
    return icon


CONCEPTS = {"A": concept_a, "B": concept_b, "C": concept_c, "D": concept_d}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "A"
    icon = CONCEPTS[choice]().resize((1024, 1024), Image.LANCZOS)
    icon.save("icon.png")
    icon.resize((512, 512), Image.LANCZOS).save("logo.png")
