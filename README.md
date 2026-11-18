# Hand Gesture Presentation System
Team Merismos (TYAIA909) · Guide: Dr. Nilima Ramteke · MIT ADT University

Control presentations with hand gestures from a normal webcam using MediaPipe hand tracking + OpenCV.

## Setup (macOS)
```bash
cd gesture-presenter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install --cask libreoffice      # only needed for .pptx/.ppt/.odp files
```
First run downloads the hand model (~7 MB). Allow camera access for your terminal
(System Settings → Privacy & Security → Camera).

## Run
```bash
python -m gesture_presenter "My Deck.pptx"   # .pptx .ppt .pps(x) .odp .key .pdf
python -m gesture_presenter                  # opens a file picker
```
PowerPoint/OpenDocument files are converted to PDF via LibreOffice (cached, so the
second open is instant). Keynote files are exported by Keynote itself.

## Gestures
| Gesture | Action |
|---|---|
| Open hand, swipe left | Next slide |
| Open hand, swipe right | Previous slide |
| Index + middle | Laser pointer |
| Index only | Draw / annotate |
| Index + middle + ring | Undo last stroke |
| Open hand held still 1.5 s | Clear annotations |
| Fist | Idle |

The status bar shows the recognised sign and which fingers (T I M R P) the app sees raised.
Keys: `f` fullscreen · `c` camera preview · `h` help panel · `a`/`d` or arrows manual nav · `q` quit.

## Design
- `slides.py` – any deck → PDF (LibreOffice / Keynote, hash-cached) → slide images (PyMuPDF); native file picker
- `tracker.py` – MediaPipe Tasks HandLandmarker (21 landmarks, VIDEO mode, one hand); finds the bundled/local/cached model
- `gestures.py` – distance-based finger test, pose table, majority-vote pose smoothing, debounce (8 stable frames), cooldown, release-to-rearm, distance-based swipes, hold-still clear, adaptive cursor smoothing. Pure Python, no OpenCV/MediaPipe, so it is fully unit tested
- `ui.py` – anti-aliased HUD (status pill, help card, toasts, pointer) drawn with Pillow
- `app.py` – per-frame pipeline split into small steps (camera thread, find hand, update pointer, render, handle key), per-slide annotation layers, camera overlay

Each module starts with named constants for its sizes, colours and thresholds.
For a full explanation of the code see [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md).

## Tests
`python -m pytest -q`
