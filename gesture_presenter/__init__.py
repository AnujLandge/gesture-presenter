"""Hand Gesture Presentation System - Team Merismos (TYAIA909), MIT ADT University.

Package layout (in pipeline order):
  slides.py   - load a presentation file as slide images
  tracker.py  - find hand landmarks in a camera image (MediaPipe)
  gestures.py - turn landmarks into poses, actions and a smooth cursor
  ui.py       - draw the on-screen HUD (status pill, help card, toasts, pointer)
  app.py      - the main loop that connects everything
"""
__version__ = "1.0.0"
