# Gravity Game - Cloth Simulation

Ein interaktives Tuch-Simulations-Spiel mit Pygame, bei dem ein physikalisch simuliertes Gitter mit dem Mauszeiger geschnitten werden kann.

## Aktueller Stand

### Features
- **Physik-Simulation**: Verlet-Integration für realistische Stoffphysik
- **Interaktion**: Schneiden des Gitters mit dem Mauszeiger
- **Visuelle Effekte**: Spannungsanzeige durch Farbänderung der Verbindungen
- **Steuerung**:
  - `ESC` - Beenden
  - `SPACE` - Gitter zurücksetzen
  - `R` - Alle Pins lösen (Gitter fällt)

### Testmodus
Für die Entwicklung gibt es einen Testmodus:
- Fenster statt Fullscreen (1200x800)
- Automatischer Timeout nach 10 Sekunden
- Aktivierung: `TEST_MODE = True` in `gravity_game.py`

### Installation

```bash
# Virtual Environment erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install pygame
```

### Ausführen

```bash
python gravity_game.py
```

---

## Nächstes Entwicklungsziel: Handsteuerung per Webcam

### Übersicht
Der Mauszeiger soll durch Handerkennung über die Webcam ersetzt werden. Die Hand des Users steuert den Cursor, und verschiedene Gesten haben unterschiedliche Effekte.

### Geplante Gesten

| Geste | Aktion |
|-------|--------|
| **Flache Handfläche** | Schneiden des Gitters |
| **Faust** | Gitter berühren/schieben (ohne zu schneiden) |

### Technische Umsetzung (Schrittweise)

#### Phase 1: Bibliothek einbinden
- [ ] **MediaPipe** von Google für Handerkennung einbinden
- [ ] OpenCV für Webcam-Zugriff installieren
- [ ] Grundlegende Webcam-Capture testen

```bash
pip install mediapipe opencv-python
```

#### Phase 2: Handerkennung implementieren
- [ ] Hand-Landmarks erkennen (21 Punkte pro Hand)
- [ ] Handposition (Mittelpunkt) als Cursor-Position nutzen
- [ ] Webcam-Feed im Hintergrund verarbeiten

#### Phase 3: Gestenerkennung
- [ ] Flache Hand erkennen (alle Finger gestreckt)
- [ ] Faust erkennen (alle Finger geschlossen)
- [ ] Gesten-Status als Variable verfügbar machen

#### Phase 4: Integration in das Spiel
- [ ] Cursor-Position durch Handposition ersetzen
- [ ] Schneiden nur bei flacher Hand aktivieren
- [ ] "Push"-Modus bei Faust (Gitter bewegen ohne schneiden)
- [ ] Visuelles Feedback für aktuelle Geste

#### Phase 5: Optimierung
- [ ] Performance-Optimierung (Threading für Kamera)
- [ ] Smoothing der Handbewegungen
- [ ] Kalibrierung für verschiedene Kamera-Positionen

### Architektur-Vorschlag

```
gravity_game/
├── gravity_game.py      # Hauptspiel (aktuell)
├── hand_tracker.py      # Neue Datei: Handerkennung
├── gestures.py          # Neue Datei: Gestenerkennung
└── config.py            # Neue Datei: Konfiguration
```

### Relevante MediaPipe-Dokumentation
- Hand Landmarks: 21 Punkte pro Hand
- Wichtige Punkte:
  - `WRIST` (0): Handgelenk
  - `THUMB_TIP` (4): Daumenspitze
  - `INDEX_FINGER_TIP` (8): Zeigefingerspitze
  - `MIDDLE_FINGER_TIP` (12): Mittelfingerspitze
  - `RING_FINGER_TIP` (16): Ringfingerspitze
  - `PINKY_TIP` (20): Kleiner Finger Spitze

### Pseudocode für Gestenerkennung

```python
def is_open_hand(landmarks):
    """Prüft ob alle Finger gestreckt sind"""
    # Vergleiche Fingerspitzen-Y mit Fingerknöchel-Y
    # Gestreckt = Spitze höher als Knöchel
    pass

def is_fist(landmarks):
    """Prüft ob Hand zur Faust geballt ist"""
    # Alle Fingerspitzen nah am Handgelenk
    pass
```

---

## Abhängigkeiten

### Aktuell
- Python 3.9+
- pygame 2.6+

### Geplant (für Handsteuerung)
- mediapipe
- opencv-python

---

## Lizenz

MIT License
