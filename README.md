# Auto Drum Transcription (ADT) -> Waveform Plotter

This repository contains a full pipeline that takes raw drum audio (like a `.wav` file) and produces a color-coded and shape-coded waveform plot showing exactly when the **Kick**, **Snare**, and **Hi-Hats** were played, even if they hit at the exact same time.

## Overview

This process is broken down into two components:
1. **Onset Detection (`src.onset`)**: Uses `librosa` to find the exact millisecond transients occur in the raw audio.
2. **Multi-Label Classification (`src.classification`)**: Uses a `scikit-learn` Random Forest to extract acoustic features (MFCCs, Spectral Centroid, ZCR) and predict an array of active instruments (`[KD, SD, HH]`).

## Getting Started

### 1. Setup Environment
Ensure you have `uv` installed, then add the core requirements:
```bash
uv add librosa matplotlib numpy scikit-learn requests tqdm
```

### 2. Fetch the Training Dataset
This project requires training a classifier using isolated drum stems. Run the fetch script to download and extract the `IDMT-SMT-Drums` dataset from Zenodo (~200MB):
```bash
uv run python src/scripts/fetch_dataset.py
```

### 3. Train the Classifier
Parse the hundreds of annotated XMLs in the dataset to build a multi-label training set and fit the model:
```bash
uv run python src/scripts/train_classifier.py
```
*This will cache a `drum_classifier.pkl` into your `data/` folder.*

### 4. Run the Pipeline!
Finally, point the main CLI at any `.wav` file to classify its hits. 

```bash
uv run python main.py <path_to_audio.wav>
```
The script will output the predictions in the console and automatically save a visual `classified_plot_<filename>.png` in your root directory!

### Plot Legend
- 🔴 **Red Circle (`o`)**: Kick Drum
- 🟢 **Green Cross (`x`)**: Snare Drum
- 🔵 **Blue Triangle (`^`)**: Hi-Hat

*(If multiple drums hit simultaneously, you will see their shapes stacked vertically on the dashed onset line.)*