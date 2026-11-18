# Code Walkthrough – Hand Gesture Presentation System

A study guide for the team. It explains what the code does, in the order it
runs, with the real numbers from the source. If this guide and the code ever
disagree, the code is right – please fix the guide.

Contents

1. How the program runs end to end
2. File-by-file tour
3. The key algorithms, explained simply
4. Likely examiner questions (with short, honest answers)
5. Glossary

---

## 1. How the program runs end to end

### Start-up (happens once)

`python -m gesture_presenter deck.pptx` runs `gesture_presenter/__main__.py`,
which calls `app.main()`. The packaged macOS app starts from `launcher.py`,
which calls the same `main()`.

`main()` then:

1. **Reads the command line** (`parse_args`): an optional file path and
   `--camera N` (default camera 0).
2. **Chooses a file** – if no path was given, `slides.pick_file()` shows a
   native file dialog. Cancelling it exits quietly.
3. **Loads the slides** (`open_presentation` → `slides.load`). Non-PDF files are
   converted to PDF first, then each PDF page becomes an image. `Presenter`
   letterboxes every slide onto a 1920 × 1080 black canvas once, up front.
4. **Opens the camera** (`open_camera`), retrying every 1 s for up to 30 s so
   the user has time to answer the macOS camera permission prompt. It asks for
   960 × 540 at 30 fps with a 1-frame buffer.
5. **Creates the helpers**: `Camera` (background reader thread),
   `HandTracker` (MediaPipe), `GestureEngine`, `Smoother`, a 1280 × 720 window
   and a `ViewState` (help card auto-hides after 15 s).
6. **Runs the loop** (`run_loop`) until the user quits, then always closes the
   camera, the tracker and the window (`finally:` block).

If anything fails (bad file, no camera), `fail()` stops the program with a
readable message. In the packaged macOS app there is no terminal, so `fail()`
also shows a macOS alert dialog.

### The 7-step per-frame pipeline (`app.run_loop`)

```
 webcam ─► 1 newest frame ─► 2 landmarks ─► 3 pose + action ─► 4 change slide
                                                     │
                                                     └──────► 5 pointer / drawing
                                                                     │
          7 show + keys ◄─ 6 draw slide, ink, pointer, HUD ◄─────────┘
```

| Step | Code | What happens |
|---|---|---|
| 1 | `Camera.latest`, `cv2.flip` | Get the newest camera image (wait up to 1 s; if none, stop). Mirror it so moving your hand right moves things right on screen. |
| 2 | `find_hand` → `HandTracker.process` | Shrink to 640 × 360, convert BGR→RGB, run MediaPipe. Result: 21 `(x, y)` points in 0..1, or `None` if no hand. |
| 3 | `GestureEngine.update` | Work out which fingers are up, look up the pose, smooth it with a majority vote, and decide whether an action fires this frame. Also auto-hides the help card when its 15 s are up. |
| 4 | `Presenter.act`, `ViewState.show_toast` | NEXT/PREV change slide, UNDO removes the last stroke, CLEAR removes all strokes on this slide. A toast ("Next slide", …) shows for 0.9 s. |
| 5 | `update_pointer` | In POINTER or DRAW pose, map the fingertip to a smoothed screen position. In DRAW pose also add the point to the current stroke. Otherwise end the stroke and reset the smoother. |
| 6 | `Presenter.render`, `overlay` | Copy the slide, draw this slide's strokes, the pointer dot and the progress bar; then the camera preview, status pill, help card and toast. |
| 7 | `cv2.imshow`, `handle_key` | Show the image, read one key: `q`/Esc quit, `c` camera preview, `h` help, `f` fullscreen, `d`/→ next, `a`/← previous. |

---

## 2. File-by-file tour

Every module starts with a docstring saying its role, then named constants
(each with a comment), then functions/classes.

### `gesture_presenter/__init__.py`, `__main__.py`, `launcher.py`
Package description and version; `__main__.py` lets `python -m
gesture_presenter` work; `launcher.py` is the script PyInstaller builds the
macOS app from (`build_mac.sh`).

### `slides.py` – file → slide images (start-up only)

| Function | What it does |
|---|---|
| `_soffice()` | Finds LibreOffice: first `soffice`/`libreoffice` on the PATH, then the usual install folders in `SOFFICE_PATHS`. |
| `_keynote_to_pdf(src, out)` | Uses AppleScript to make Keynote export a PDF (macOS only, 180 s timeout). |
| `_file_hash(path)` | First 16 hex characters of the SHA-1 of the file's bytes. |
| `_libreoffice_to_pdf(path, out)` | Runs LibreOffice headless in a temporary folder with its own profile (so it works even if LibreOffice is already open), 300 s timeout, then moves the PDF into the cache. |
| `to_pdf(path)` | Checks the extension is in `SUPPORTED`; PDFs are returned as-is; other files are converted once and cached as `~/.cache/gesture-presenter/<hash>.pdf`. |
| `load(path, zoom=2.0)` | Opens the PDF with PyMuPDF, renders each page at 2× zoom, converts RGB→BGR for OpenCV. Errors if there are no pages. |
| `pick_file()` | File dialog: AppleScript on macOS, tkinter elsewhere; `None` if cancelled or unavailable. |
| `fit(img, w, h)` | Letterbox: scale the slide to fit inside w × h without stretching, centre it on black. |

### `tracker.py` – image → 21 landmarks

| Name | What it does |
|---|---|
| `_find_model_path()` / `MODEL_PATH` | Looks for `hand_landmarker.task` (1) inside the packaged app (`sys._MEIPASS/gesture_presenter/models/`), (2) in `gesture_presenter/models/`, else (3) uses `~/.cache/gesture-presenter/` as the download location. |
| `ensure_model()` | Downloads the ~7 MB model from Google's storage the first time if it is missing. |
| `HandTracker.__init__` | Creates a MediaPipe HandLandmarker in VIDEO mode, `num_hands=1`, confidence thresholds 0.5. |
| `HandTracker._next_timestamp_ms` | Milliseconds since start, forced to always increase (VIDEO mode requires it). |
| `HandTracker.process(rgb)` | Runs the model and returns 21 `(x, y)` tuples for the first hand, or `None`. |
| `HandTracker.close` | Frees the model. |

### `gestures.py` – landmarks → pose, action, cursor (pure Python)

| Name | What it does |
|---|---|
| Landmark constants | `WRIST = 0`, `THUMB_TIP = 4`, `INDEX_TIP = 8`, `MIDDLE_MCP = 9`, … (MediaPipe numbering). |
| `Gesture` | Enum of poses: NONE, NEXT, PREV, POINTER, DRAW, UNDO, PALM, CLEAR, FIST. |
| `DISCRETE`, `CONTINUOUS` | NEXT/PREV/UNDO fire once; POINTER/DRAW act every frame. |
| `POSE_TABLE` | Finger pattern → pose (see section 3). |
| `_thumb_is_up`, `_finger_is_up`, `fingers_up` | The distance-based finger test; returns `[thumb, index, middle, ring, pinky]` as 0/1. |
| `classify(lm)` | One-frame pose lookup (used by tests). |
| `Config` | All engine thresholds in one place. |
| `GestureEngine.update(lm, now)` | The per-frame entry point; returns `(steady_pose, fired_action_or_None)`. Also stores `fingers` for the status pill. |
| `_reset_when_hand_lost` | No hand: clear history and swipe track, re-arm, fingers all down. |
| `_pose` | Majority vote over the last 5 raw poses. |
| `_update_open_hand` | PALM pose: swipe detection, then still-hold detection. |
| `_detect_swipe`, `_detect_still_hold` | Return NEXT/PREV or CLEAR when their rule is met. |
| `_update_finger_pose` | Every other pose: re-arm on non-action poses; fire DISCRETE poses after the debounce. |
| `_after_fire` | Start the cooldown, disarm, clear swipe/hold state. |
| `Smoother.map(x, y, w, h)` | Camera position → smoothed screen pixel. `reset()` forgets the position. |

### `ui.py` – heads-up display

| Name | What it does |
|---|---|
| `S`, `px(v)` | UI scale 1.5: a "design size" of 16 becomes 24 pixels on the 1920 × 1080 canvas. |
| `font(size, bold)` | Loads the first system font that exists (macOS, Windows, Linux list); cached. |
| `_panel`, `_bgra` | Make a rounded, semi-transparent dark panel; convert Pillow RGBA → OpenCV BGRA. |
| `status_pill(...)` | Bottom-left bar: gesture name, "page / total", and five finger dots T I M R P (lit = seen raised). Cached (256 entries). |
| `help_card()` | List of gestures and keys. Drawn once and cached. |
| `toast(text)` | Centred pop-up text. Cached per text. |
| `blit(frame, img, x, y)` | Alpha-blends a BGRA panel onto the BGR frame, kept inside the frame. |
| `pointer(frame, p, drawing)` | Red laser dot with a soft glow (radius 6 design points, 4 while drawing) and a white ring. |

### `app.py` – the main loop

| Name | What it does |
|---|---|
| `W, H`, constants | Canvas size 1920 × 1080 and every size/colour/time used by the app. |
| `LABEL`, `HAND_LINKS` | Text for each gesture; landmark pairs that form the hand skeleton drawing. |
| `Presenter` | Slides, current index `i`, strokes per slide, `drawing` flag. `go`, `act`, `draw_point`, `render`. |
| `_draw_camera_preview`, `overlay` | Camera preview with skeleton (top right), status pill, help card, toast. |
| `Camera` | Background thread that keeps only the newest frame; `latest`, `close`. |
| `open_camera`, `fail` | Open the webcam with retries; exit with a message/dialog. |
| `ViewState` | Camera/help/fullscreen switches and the toast text with its expiry time. |
| `parse_args`, `open_presentation`, `create_window` | Start-up helpers. |
| `find_hand`, `update_pointer` | Steps 2 and 5 of the pipeline. |
| `smooth_stroke`, `fit_to_window` | Tidy a finished stroke; pad the frame to the window shape in fullscreen. |
| `toggle_fullscreen`, `handle_key` | Step 7. |
| `run_loop`, `main` | The pipeline loop and the program entry point. |

### `tests/test_gestures.py`
21 pytest tests for `gestures.py`. `hand(up, cx, wrist_y, angle)` builds a
fake hand with chosen fingers raised, at a chosen position and rotation, so
every rule can be tested without a camera.

---

## 3. The key algorithms, explained simply

### 3.1 Finger test (is each finger up?)

MediaPipe numbers each finger's joints from the knuckle (MCP) to the tip. For
the **index, middle, ring and pinky** fingers:

> A finger is **up** if `distance(tip, wrist) > 1.12 × distance(PIP joint, wrist)`.

A straight finger's tip is well beyond its middle joint; a curled finger folds
its tip back towards the wrist. The **12% margin** stops a half-bent finger
from flickering between up and down.

**Why distances and not "tip is above the joint" (y-coordinates)?** A
y-coordinate test only works when the hand points straight up. Distances do not
change when the hand rotates, so the test works for tilted or upside-down hands
and for left or right hands. The tests check angles from −1.2 to 3.14 radians.

The **thumb** bends sideways, so it has its own rule. It is up only if both:
- `distance(thumb tip, pinky knuckle) > 1.15 × distance(thumb IP joint, pinky knuckle)` (it sticks out away from the palm), and
- `distance(thumb tip, index knuckle) > 0.45 × palm length` (it is not tucked
  against the index finger), where palm length = distance(wrist, middle knuckle).

### 3.2 Pose table

Finger pattern `(thumb, index, middle, ring, pinky)` → pose:

| Pattern | Pose | Meaning |
|---|---|---|
| 1 0 0 0 0 | PREV | thumb only → previous slide |
| 0 0 0 0 1 | NEXT | pinky only → next slide |
| x 1 0 0 0 | DRAW | index only (thumb either way) |
| x 1 1 0 0 | POINTER | index + middle |
| x 1 1 1 0 | UNDO | three fingers |
| x 1 1 1 1 | PALM | open hand (swipe / hold still) |
| 0 0 0 0 0 | FIST | rest, nothing happens |
| anything else | NONE | "Unknown sign" |

`x` = thumb up or down both accepted, because the thumb is the hardest finger
to read reliably. The help card advertises swipes for slide changes; the
thumb-only and pinky-only poses are an extra way to do the same thing.

### 3.3 Majority vote (steady pose)

The engine remembers the raw pose of recent frames. The **reported pose** is
the one that appears most often in the **last 5 frames**; on a tie, the most
recent of the tied poses wins. So one misread frame (or two) cannot change the
pose, and the drawing stroke is not broken by a single bad frame (there is a
test for exactly this). Because the history needs at least 3 of the last 5
frames to agree for a clear majority, a real change of pose shows after about
3 frames (≈ 0.1 s at 30 fps).

### 3.4 Debounce for one-shot actions (NEXT, PREV, UNDO poses)

An action fires only when **all** of these are true:

1. **Stable:** the raw pose was the same in each of the last **8 frames**
   (≈ 0.27 s at 30 fps). This is stricter than the vote on purpose: a wrong
   slide change is much more annoying than a slightly late one.
2. **Cooldown:** at least **1.2 s** since the last fired action.
3. **Armed:** after any action fires, the engine is *disarmed*. It re-arms only
   when you show a non-action pose (fist, pointer, draw or an unknown sign) or
   take your hand out of view. The open hand does **not** re-arm it.

Result: holding the pinky up for 3 seconds gives **one** "Next slide", not ten.

### 3.5 Swipe (open hand moving sideways)

While the pose is PALM (and the cooldown has passed), the engine records the
position of the middle-finger knuckle with a timestamp and keeps only the last
**0.9 s**. A swipe fires when, compared with the oldest kept point:
- the hand moved at least **30% of the camera frame width** horizontally, and
- vertical movement is **less than 0.6 ×** the horizontal movement (mostly sideways).

Because the camera image is mirrored, moving left gives a negative change in x
→ **NEXT**; moving right → **PREV**. A swipe does not need the 8 stable frames.
During the 1.2 s cooldown the track is cleared, which is what stops the hand's
**return stroke** from firing the opposite swipe.

### 3.6 Clear (open hand held still)

If the open hand is not swiping and the engine is armed, it remembers where
the hand was when it became still. If the hand moves more than **0.06** (6% of
the frame) away, the timer restarts there. If it stays within that distance
for **1.5 s**, CLEAR fires and all drawings on the current slide are removed.

### 3.7 Adaptive cursor smoothing (`Smoother`)

1. **Margin mapping:** the outer **12%** of the camera image on each side is
   ignored and the middle 76% is stretched to the full screen, so you can reach
   the screen corners without your hand leaving the camera view.
2. **Dead zone:** if the new target is **3 px** or less from the current cursor,
   the cursor does not move. This hides natural hand tremor.
3. **Adaptive exponential smoothing:** otherwise the cursor moves a fraction
   `alpha` of the way to the target:
   `alpha = 0.30 + (0.8 − 0.30) × min(distance / 90 px, 1)`.
   Slow, small movements → alpha near 0.30 → very steady (good for pointing at
   a word). Fast movements of 90 px or more → alpha 0.8 → little lag.
4. When the pose leaves POINTER/DRAW, `reset()` is called so the next time the
   cursor appears it jumps straight to the hand.

The index fingertip drives the cursor in both poses. POINTER (two fingers)
moves the cursor with the "pen up"; DRAW (index only) puts the pen down at the
same spot, which is what lets you write separate letters. While drawing, a new
point is added to the stroke only if it moved at least **3 px** (sum of x and y
change) from the last point. When the pen lifts, `smooth_stroke` tidies the
finished stroke with a 5-point moving average (first and last points are kept).

### 3.7b Fullscreen padding (`fit_to_window`)

The canvas is 16:9 but a MacBook screen is 16:10, so fullscreen leaves spare
space. `fit_to_window` asks OpenCV for the window size, works out how many
extra rows (or columns) are needed to match the window shape, and adds them
equally on both sides in the slide's own edge colour. Result: the slide is
centred and no black band shows.

### 3.8 Camera thread

`cv2.VideoCapture.read()` returns frames in order from a queue. While the main
loop is busy (MediaPipe + drawing a 1920 × 1080 image), new frames pile up, so
the loop would process older and older frames and the pointer would lag behind
the real hand. The `Camera` class reads frames continuously on a separate
**daemon thread** and keeps only the newest one (protected by a lock). Each
frame gets a sequence number so the main loop never processes the same frame
twice. If no new frame arrives within 1 s, the loop ends.

### 3.9 PDF conversion and the hash cache

OpenCV cannot read PowerPoint files, but PyMuPDF renders PDFs quickly. So:
- `.pdf` → used directly.
- `.key` on macOS → Keynote exports a PDF.
- everything else (`.pptx .ppt .ppsx .pps .odp`, and `.key` off macOS) →
  LibreOffice converts to PDF in headless mode.

The PDF is saved as `~/.cache/gesture-presenter/<first 16 hex chars of SHA-1>.pdf`.
The name comes from the file's **contents**, so opening the same deck again
skips the slow conversion, while an edited deck (different bytes) is converted
again.

---

## 4. Likely examiner questions

**Q1. Why MediaPipe instead of training your own CNN?**
MediaPipe's hand landmark model is pre-trained by Google on a large hand
dataset and runs in real time on a laptop CPU. Training our own detector would
need a large labelled dataset and a GPU, and would very likely be less accurate.
We focused our work on what to do with the landmarks.

**Q2. Then what did you build yourselves?**
Everything after the landmarks: the rotation-invariant finger test, the pose
table, the majority vote, the debounce/cooldown/re-arm logic, swipe and hold
detection, the adaptive cursor smoother, per-slide drawing, the HUD, the
camera thread and the file conversion pipeline, plus the unit tests.

**Q3. Why hand-written rules instead of a machine-learning classifier on the landmarks?**
Our gestures are defined by which fingers are raised, which a simple geometric
rule captures directly. Rules need no training data, are fully explainable, and
can be unit-tested with exact inputs. The downside is that adding very
different gestures (e.g. sign-language letters) would need an ML classifier.

**Q4. What are the 21 landmarks?**
The wrist plus four points on each of the five fingers (knuckle to tip), in a
fixed MediaPipe numbering. We use x and y, normalised to 0..1 of the image size.

**Q5. What happens with two hands in view?**
The tracker is created with `num_hands=1`, so MediaPipe reports at most one
hand and the program follows that one. We do not choose which hand; MediaPipe
does. Two-hand gestures are not supported.

**Q6. Does it work for left-handed users?**
Yes. The finger test uses distances only, so it is the same for either hand.

**Q7. Why the majority vote AND the 8 stable frames?**
They do different jobs. The vote (3 of 5) keeps the displayed pose and the
pointer/drawing steady with little delay. The 8-frame rule is a stricter check
only for actions that change slides or undo, where a mistake is costly.

**Q8. Why a cooldown and "release to re-arm"?**
Without them, holding a pose would fire every frame (30 times a second).
Re-arming means "one gesture = one action". The cooldown also stops the hand's
return movement after a swipe being read as a second swipe.

**Q9. Why is the camera image mirrored?**
So the screen behaves like a mirror: moving your hand to the right moves the
pointer to the right. It also makes swipe directions feel natural.

**Q10. Why run the camera on a separate thread?**
See 3.8: to always process the newest frame and avoid growing lag.

**Q11. How does the pointer stay steady but still responsive?**
See 3.7: dead zone for tremor, heavy smoothing for slow moves, light smoothing
for fast moves.

**Q12. How did you test it?**
21 automated unit tests (`pytest`) on `gestures.py`, using synthetic hands at
different positions and rotations. They cover the finger test, pose table,
rotation invariance, debounce, cooldown, re-arming, swipes in both directions,
ignoring the return stroke, slow drift and vertical movement not being swipes,
the clear-hold, recovery from a misread frame, and the cursor smoother. The
camera, MediaPipe and the display were tested by hand, by using the app; we do
not have automated tests or measured accuracy figures for those parts.

**Q13. Why is `gestures.py` free of OpenCV and MediaPipe?**
So the logic can be tested quickly without a camera or the model, and so the
recognition rules are separate from the input source.

**Q14. Which libraries do you use and for what?**
- **MediaPipe** – hand landmark detection (pre-trained model).
- **OpenCV (cv2)** – camera capture, resizing, colour conversion, drawing lines/circles, the window and keyboard.
- **NumPy** – images are NumPy arrays; alpha blending maths.
- **PyMuPDF** – rendering PDF pages to images.
- **Pillow** – smooth (anti-aliased) text and rounded panels for the HUD.
- **LibreOffice / Keynote** (external programs) – converting decks to PDF.
- Standard library: `argparse`, `threading`, `hashlib`, `subprocess`, `tempfile`, `urllib`.

**Q15. What are the limitations?**
- Needs reasonable lighting and the hand clearly visible to the webcam.
- One hand only; fixed gesture set.
- Thresholds (12%, 30%, 1.2 s, …) were chosen by hand and are not auto-tuned
  per user or camera.
- PowerPoint animations, transitions, video and speaker notes are lost because
  slides are shown as static PDF images.
- Opening PowerPoint files needs LibreOffice installed.
- No automated tests for the camera/display parts and no measured accuracy figures.

**Q16. What happens if the hand leaves the camera?**
`update` receives `None`: history and swipe track are cleared, the engine is
re-armed, the status shows "Show your hand to the camera", and any stroke ends.

**Q17. How do you prevent accidental slide changes while drawing or pointing?**
Pointer and draw poses are not action poses, so they never change slides. A
slide change needs a deliberate swipe (30% of the frame width in 0.9 s, mostly
sideways) or a pose held for 8 identical frames, plus the cooldown and re-arm rules.

**Q18. Are drawings kept per slide?**
Yes. `Presenter.strokes` maps slide number → list of strokes. Going back to a
slide shows its drawings again. Undo and clear only affect the current slide.
Drawings are not saved when the program closes.

**Q19. Why convert everything to PDF?**
PDF is a single format that PyMuPDF renders reliably and quickly, so the rest of
the program only has to handle one input type. Conversion is cached by hash.

**Q20. How fast is it?**
The camera is asked for 30 fps, and MediaPipe gets a reduced 640 × 360 image to
keep it fast. HUD panels are cached so they are not redrawn each frame. We have
not published measured frame-rate numbers; the real rate depends on the
computer and camera.

**Q21. Why the "magic" numbers like 1.12 or 0.6?**
They are named constants at the top of each file, with comments. They were
chosen by trying the app and adjusting until it felt reliable; the unit tests
lock in the behaviour they produce.

**Q22. Does any video leave the computer?**
No. Processing is local. The only network access is the one-time download of
the hand model file if it is not already present (the packaged app bundles it).

---

## 5. Glossary

- **Landmark** – one of 21 named points on the hand found by MediaPipe.
- **Normalised coordinates** – x and y as fractions (0..1) of the image width and height.
- **MCP / PIP / IP** – finger joints: knuckle (MCP), middle joint (PIP), and the thumb's single middle joint (IP).
- **Pose** – the hand shape in a frame (e.g. DRAW = index finger up).
- **Discrete action** – fires once (next, previous, undo, clear).
- **Continuous pose** – acts every frame while held (pointer, draw).
- **Majority vote** – choosing the value that appears most often in a small window.
- **Debounce** – requiring a signal to be stable for a while before acting on it.
- **Cooldown** – minimum time between two actions.
- **Re-arm** – allowing an action again after the hand has been "released".
- **Exponential smoothing** – moving a fraction (alpha) of the way to the new value each frame.
- **Dead zone** – movements smaller than this are ignored.
- **Letterbox** – fitting an image into a screen without stretching, padding with black.
- **BGR / RGB / BGRA** – colour channel orders; OpenCV uses BGR, MediaPipe and Pillow use RGB(A); the A is transparency (alpha).
- **Alpha blending** – mixing two images using transparency: `result = top × a + bottom × (1 − a)`.
- **Daemon thread** – a background thread that does not keep the program running when the main thread ends.
- **Lock** – makes sure two threads do not read/write the same data at the same moment.
- **SHA-1 hash** – a short fingerprint of a file's contents; same bytes → same hash.
- **Headless** – running a program (LibreOffice) without opening its window.
- **VIDEO running mode** – MediaPipe mode that tracks the hand between consecutive frames instead of searching from scratch each time.
- **HUD** – heads-up display: the overlays drawn on top of the slide.
- **PyInstaller / `sys._MEIPASS`** – the tool that builds the macOS app; `_MEIPASS` is the folder where the packaged app's files are unpacked at run time.
