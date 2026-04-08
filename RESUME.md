# Project Resumption: GMD Migration & Full Kit Transcription

This file contains the current state of the Automatic Drum Transcription (ADT) project to allow for seamless resumption.

## Current Goal
Migrate the training pipeline from the small IDMT-SMT-Drums dataset to the **Groove MIDI Dataset (GMD)** and expand transcription support to the **full drum kit** (Kick, Snare, Hi-Hat, Toms, and Cymbals).

## Project Status

### ✅ Completed
- **Architecture Update**: Expanded `DRUM_CLASSES` in `src/classification.py` to `['KD', 'SD', 'HH', 'TT', 'CY']`.
- **Plotting Pipeline**: Updated `main.py` with specific colors/shapes for Toms and Cymbals and implemented dynamic vertical stacking for simultaneous hits.
- **Dataset Logic**: Rewrote `src/scripts/fetch_dataset.py` to target the GMD audio/MIDI zip (~26GB).
- **Training Pipeline**: Completely overhauled `src/scripts/train_classifier.py` to use `mido` for parsing GMD's MIDI ground truth and mapping General MIDI notes to our classes.
- **Environment Setup**: Added `mido` to the project dependencies and synced the local virtualenv.
- **Model Artifact**: Rebuilt `data/drum_classifier.pkl` from GROOVE data using the updated full-kit pipeline.
- **Smoke Test**: Verified `main.py` on `RealDrum01_00#MIX.wav` and confirmed plot generation.
- **Documentation**: Updated `README.md` with new training instructions and the updated plot legend.

### ⚠️ In Progress / Errors to Address
- **Optional Full Retrain**: The trainer now supports `GMD_MAX_TRACKS` and `GMD_SPLITS` for faster local verification. Unset those variables for a full corpus pass.

### ⏭️ Next Steps
1. **Run Evaluation**: Use `uv run python src/scripts/test_suite.py` to compare the updated model against the bundled annotation XML files.
2. **Optional Full Retrain**: Re-run `uv run python src/scripts/train_classifier.py` without `GMD_MAX_TRACKS` if you want the complete GROOVE corpus instead of the smoke-tested slice.

---
*Status recorded at: 2026-04-08 03:43 AM*
