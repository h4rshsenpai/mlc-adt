# Auto Drum Transcription

This project detects drum hit onsets in audio, classifies each hit as one or more drum instruments, and renders the result as an annotated waveform plot. It is built around a lightweight classical ML pipeline rather than a deep-learning stack, which makes it easier to inspect, retrain, and iterate on locally.

The current label set is:

- `KD`: kick drum
- `SD`: snare drum
- `HH`: hi-hat
- `TT`: toms
- `CY`: cymbals

Because the classifier is multi-label, a single onset can be tagged with more than one instrument when hits occur simultaneously.

## Quick Start

### Requirements

- Python `>=3.14` as declared in [pyproject.toml](/Users/xrisk/dev/mlc-adt/pyproject.toml)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Enough local disk for the Groove MIDI Dataset if you plan to train the model; the archive is roughly 26 GB before extraction

### Install dependencies

```bash
uv sync
```

### Important note for `uv` cache placement

In some environments, especially sandboxed runs or systems with a read-only home directory, `uv` can fail unless its cache directory points at a writable location. In those cases, set `UV_CACHE_DIR` before running `uv` commands.

Example:

```bash
export UV_CACHE_DIR=.uv-cache
uv sync
```

You can use the same pattern for any project command:

```bash
UV_CACHE_DIR=.uv-cache uv run python main.py path/to/audio.wav
```

### Run inference with an existing model

The main entrypoint expects a trained model at `data/drum_classifier.pkl`.

```bash
uv run python main.py path/to/audio.wav --zoom 0:10
```

This prints the predicted hits to the terminal and writes a plot like `classified_plot_<parent>_<filename>.png` into the repository root.

## End-to-End Workflow

### 1. Download the training dataset

This project trains on the Groove MIDI Dataset (GMD) from Google Magenta. The downloader stores the archive at `data/groove-v1.0.0.zip` and extracts it under `data/groove/`.

```bash
uv run python src/scripts/fetch_dataset.py
```

If you already have the dataset extracted, place it at `data/groove/` and skip this step.

### 2. Train the classifier

```bash
uv run python src/scripts/train_classifier.py
```

Training parses MIDI annotations, groups near-simultaneous hits into a single multi-label target, extracts per-hit audio features, fits a `RandomForestClassifier`, prints a classification report, and saves the trained model to `data/drum_classifier.pkl`.

Useful environment variables for smaller development runs:

- `GMD_MAX_TRACKS`: cap the number of GMD tracks processed
- `GMD_SPLITS`: comma-separated split filter such as `train,test`

Example:

```bash
GMD_MAX_TRACKS=50 GMD_SPLITS=train UV_CACHE_DIR=.uv-cache uv run python src/scripts/train_classifier.py
```

### 3. Evaluate a saved model

```bash
uv run python src/scripts/evaluate.py
```

This rebuilds the dataset, recreates the held-out split, loads the saved model, and prints per-class plus aggregate metrics.

### 4. Run the test suite helper

```bash
uv run python src/scripts/test_suite.py
```

This script runs the full inference pipeline against a few fixed audio cases and, where XML annotations are available, reports onset F1 and exact class-match accuracy.

## Technical Overview

### Pipeline summary

The system has two main stages:

1. Onset detection finds candidate drum hits in the waveform.
2. Multi-label classification predicts which drum classes were active at each detected onset.

The inference entrypoint lives in [main.py](/Users/xrisk/dev/mlc-adt/main.py). It orchestrates model loading, onset detection, classification, console output, and waveform rendering.

### Onset detection

Implemented in [src/onset.py](/Users/xrisk/dev/mlc-adt/src/onset.py).

Key ideas:

- Audio is loaded with `librosa.load(..., sr=None)` to preserve the source sample rate.
- Tempo is estimated with `librosa.beat.beat_track`.
- The minimum inter-onset interval is derived from the estimated tempo, using half of a 16th-note duration as a refractory window.
- `librosa.onset.onset_strength` and `librosa.onset.onset_detect` are then used to convert the waveform into onset frame indices and timestamps.

This tempo-aware wait window is intended to suppress retriggers caused by ringing and bleed while still allowing fast drum passages.

### Feature extraction and classification

Implemented in [src/classification.py](/Users/xrisk/dev/mlc-adt/src/classification.py).

For each onset, the classifier looks at a 100 ms slice after the hit and extracts:

- 20 MFCCs from a high-pass filtered signal
- spectral centroid
- spectral rolloff
- spectral bandwidth
- zero-crossing rate
- time since the previous onset

These features are designed to separate low-frequency kick energy from brighter and noisier percussion content while also giving the model short-term rhythmic context.

The saved model is a scikit-learn `RandomForestClassifier` trained as a multi-output binary predictor over the five drum labels.

### Training data construction

Implemented in [src/scripts/train_classifier.py](/Users/xrisk/dev/mlc-adt/src/scripts/train_classifier.py).

Training works by aligning GMD MIDI events with the matching audio files:

- MIDI notes are mapped into the project label space (`KD`, `SD`, `HH`, `TT`, `CY`).
- Events within a 30 ms tolerance are grouped into a single multi-label target.
- Matching audio windows are sliced from the waveform.
- Features are extracted per grouped onset and paired with multi-hot label vectors.

This produces a supervised dataset suitable for classical multi-label classification.

### Plot generation

The output plot is generated in [main.py](/Users/xrisk/dev/mlc-adt/main.py). Each onset is drawn as a dashed vertical line, and predicted classes are stacked vertically with distinct colors and markers so simultaneous hits remain visible.

Current plot legend:

- red circle: `KD`
- green `X`: `SD`
- blue triangle: `HH`
- yellow square: `TT`
- magenta diamond: `CY`

## Repository Layout

```text
.
├── main.py
├── pyproject.toml
├── src/
│   ├── classification.py
│   ├── onset.py
│   └── scripts/
│       ├── evaluate.py
│       ├── fetch_dataset.py
│       ├── test_suite.py
│       └── train_classifier.py
└── data/
    ├── drum_classifier.pkl
    └── groove/
```

## Project Organization

The repository is intentionally small and split by responsibility:

- [main.py](/Users/xrisk/dev/mlc-adt/main.py) is the user-facing CLI entrypoint for inference and plot generation.
- [src/onset.py](/Users/xrisk/dev/mlc-adt/src/onset.py) owns onset detection and waveform-level preprocessing for hit timing.
- [src/classification.py](/Users/xrisk/dev/mlc-adt/src/classification.py) owns feature extraction and per-onset multi-label prediction.
- [src/scripts/train_classifier.py](/Users/xrisk/dev/mlc-adt/src/scripts/train_classifier.py) owns dataset construction and model training.
- [src/scripts/evaluate.py](/Users/xrisk/dev/mlc-adt/src/scripts/evaluate.py) evaluates a saved model without retraining it.
- [src/scripts/test_suite.py](/Users/xrisk/dev/mlc-adt/src/scripts/test_suite.py) runs a small set of end-to-end checks against known audio inputs.
- [src/scripts/fetch_dataset.py](/Users/xrisk/dev/mlc-adt/src/scripts/fetch_dataset.py) downloads and extracts the external training corpus.

A few practical conventions are worth knowing:

- Source code lives under `src/`, while one-off operational entrypoints live in `src/scripts/`.
- Large generated assets and external datasets are expected under `data/` rather than committed into the source tree.
- The trained model is treated as a runtime artifact, not hand-edited source.
- Plots are generated into the repository root today, which keeps inference outputs easy to inspect but does mean repeated runs accumulate image files there.

## Output Artifacts

- `data/drum_classifier.pkl`: trained scikit-learn model
- `classified_plot_*.png`: waveform plot annotated with predicted drum labels
- `data/groove/`: extracted Groove MIDI Dataset

## Operational Notes

- [src/onset.py](/Users/xrisk/dev/mlc-adt/src/onset.py) already forces writable cache locations for Matplotlib and related tooling because some execution environments make the default cache path read-only.
- `UV_CACHE_DIR` is separate from Matplotlib or XDG cache handling; it specifically helps `uv` itself use a writable cache directory.
- Training and evaluation can take a while on the full dataset, and the first pass is I/O-heavy because every track must be loaded and paired with its MIDI metadata.

## Known Limitations

- The pipeline assumes isolated or drum-dominant audio and has not been hardened for dense full-mix transcription.
- Evaluation scripts rebuild features from scratch rather than caching an intermediate training dataset.
- Inference quality depends heavily on onset detection quality; missed or duplicated onsets cannot be corrected by the classifier stage.
