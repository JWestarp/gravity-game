# Gravity Game - Cloth Simulation

An interactive cloth simulation game using Pygame, where a physically simulated grid can be cut with hand gestures tracked via webcam.

## Current Status

### Features
- **Physics Simulation**: Verlet integration for realistic cloth physics
- **Hand Tracking**: Control cursor with hand gestures via webcam
- **Gesture Recognition**: Point finger to cut, make fist to push without cutting
- **Visual Effects**: Tension display through color changes in connections
- **Hand Skeleton Overlay**: Schematic display of tracked hand
- **Controls**:
  - `ESC` - Exit
  - `SPACE` - Reset grid
  - `R` - Release all pins (grid falls)

### Test Mode
For development, there is a test mode:
- Window instead of fullscreen (1200x800)
- Automatic timeout after 60 seconds
- Activation: `TEST_MODE = True` in `gravity_game.py`

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pygame mediapipe opencv-python
```

### Running

```bash
python gravity_game.py
```

---

## Hand Tracking Features

### Supported Gestures

| Gesture | Action |
|---------|--------|
| **Pointing finger** | Cut the grid |
| **Fist** | Touch/push grid (without cutting) |

### Technical Implementation

The hand tracking uses:
- **MediaPipe** by Google for hand landmark detection
- **OpenCV** for webcam access
- **21 hand landmarks** tracked per hand
- **Smoothing algorithm** for fluid cursor movement

### Architecture

```
gravity_game/
├── gravity_game.py      # Main game
├── hand_tracker.py      # Hand detection module
├── hand_landmarker.task # MediaPipe model (auto-downloaded)
├── start.sh             # Start script
└── README.md            # Documentation
```

### MediaPipe Hand Landmarks
- Key points:
  - `WRIST` (0): Wrist
  - `THUMB_TIP` (4): Thumb tip
  - `INDEX_FINGER_TIP` (8): Index finger tip (used for cursor)
  - `MIDDLE_FINGER_TIP` (12): Middle finger tip
  - `RING_FINGER_TIP` (16): Ring finger tip
  - `PINKY_TIP` (20): Pinky tip

---

## Configuration

Key settings in `gravity_game.py`:

```python
# Test mode configuration
TEST_MODE = True              # Window mode for testing
TEST_WINDOW_SIZE = (1200, 800)
TEST_TIMEOUT_SECONDS = 60

# Hand tracking
USE_HAND_TRACKING = True      # Enable/disable hand tracking
SHOW_HAND_OVERLAY = True      # Show hand skeleton overlay
HAND_OVERLAY_ALPHA = 0.25     # Overlay transparency

# Physics
SPACING = 25                  # Grid spacing
GRAVITY = 0.8                 # Gravity strength
STIFFNESS = 5                 # Constraint iterations
FRICTION = 0.99               # Movement friction
CUT_RADIUS = 30               # Cutting radius
```

---

## Dependencies

- Python 3.9+
- pygame 2.6+
- mediapipe 0.10+
- opencv-python

---

## Credits

Original cloth simulation code by **Mohsin Ali** ([@MOHCINALE](https://github.com/MOHCINALE))

Hand tracking integration and enhancements by the current maintainers.

---

## License

MIT License
