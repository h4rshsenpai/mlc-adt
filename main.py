import sys
import os
import argparse
import pickle
import numpy as np

from src.onset import detect_onsets
from src.classification import classify_hits

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "drum_classifier.pkl")

def generate_classification_plot(y, sr, onset_times, predictions, output_path, title, zoom_range=None):
    """
    Plots the waveform and color-codes/shape-codes the hits based on multi-label classification.

    Args:
        zoom_range: Optional (start_sec, end_sec) tuple to restrict the x-axis view.
                    Defaults to None (full track).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa.display

    t_start, t_end = zoom_range if zoom_range else (None, None)

    print(f"Plotting multi-label onsets to {output_path}...")
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)

    # Define distinct colors and marker shapes for our classes
    style_map = {
        'KD': {'color': 'r', 'marker': 'o'},  # Red Circle
        'SD': {'color': 'g', 'marker': 'X'},  # Green Cross
        'HH': {'color': 'b', 'marker': '^'}   # Blue Triangle
    }
    labels_plotted = set()

    for t, active_classes in zip(onset_times, predictions):
        if not active_classes:
            continue
        # Skip onsets outside the zoom window
        if t_start is not None and t < t_start:
            continue
        if t_end is not None and t > t_end:
            continue

        # Draw a generic dashed line to mark the onset
        plt.vlines(t, ymin=-1, ymax=1, color='k', alpha=0.3, linestyle='--')

        # Stack markers vertically to represent simultaneous hits
        num_classes = len(active_classes)
        if num_classes == 1:
            y_offsets = [0.8]
        elif num_classes == 2:
            y_offsets = [0.8, -0.8]
        else:
            y_offsets = [0.8, 0.0, -0.8]

        for i, cls in enumerate(active_classes):
            c = style_map[cls]['color']
            m = style_map[cls]['marker']
            label = cls if cls not in labels_plotted else None
            plt.scatter(t, y_offsets[i], color=c, marker=m, s=100, label=label, zorder=5)
            if label:
                labels_plotted.add(cls)

    if zoom_range:
        plt.xlim(t_start, t_end)
        title += f" ({t_start}s – {t_end}s)"

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def parse_zoom(zoom_str):
    """Parse a 'start:end' zoom string into a (float, float) tuple."""
    try:
        parts = zoom_str.split(':')
        if len(parts) != 2:
            raise ValueError
        return float(parts[0]), float(parts[1])
    except (ValueError, AttributeError):
        print(f"ERROR: Invalid --zoom format '{zoom_str}'. Expected 'start:end', e.g. '10:20'")
        sys.exit(1)

def main(audio_path, zoom_range=None):
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Run `python src/scripts/train_classifier.py` first.")
        sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    # Phase 1: Onset Detection (tempo-aware refractory window)
    y, sr, onset_samples, onset_times, bpm = detect_onsets(audio_path)
    print(f"[{os.path.basename(audio_path)}] Estimated tempo: {bpm:.1f} BPM")

    if len(onset_samples) == 0:
        print("No onsets detected.")
        return

    # Load Model
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)

    # Phase 2: Classification (Multi-Label)
    print(f"[{os.path.basename(audio_path)}] Extracting features and classifying hits...")
    predictions = classify_hits(clf, y, sr, onset_samples)

    # Print Results
    print("\n--- MULTI-LABEL CLASSIFICATION RESULTS ---")
    for t, active_classes in zip(onset_times, predictions):
        if active_classes:
            labels_str = " + ".join(active_classes)
            print(f"[{t:.3f}s] -> {labels_str}")

    # Generate a unique plot filename that includes parent folder to avoid collisions
    parent = os.path.basename(os.path.dirname(os.path.abspath(audio_path)))
    basename = os.path.basename(audio_path)
    safe_name = f"{parent}_{basename}" if parent not in (".", "") else basename
    plot_path = os.path.join(PROJECT_ROOT, f"classified_plot_{safe_name}.png")

    generate_classification_plot(
        y, sr, onset_times, predictions, plot_path,
        f"Classified Waveform for {os.path.basename(audio_path)}",
        zoom_range=zoom_range,
    )
    print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADT: Automatic Drum Transcription")
    parser.add_argument("audio", nargs="?",
                        default=os.path.join(PROJECT_ROOT, "RealDrum01_00#MIX.wav"),
                        help="Path to the drum audio .wav file")
    parser.add_argument("--zoom", default="0:10",
                        help="Time range to display in the plot, as 'start:end' in seconds (default: 0:10)")
    args = parser.parse_args()
    main(args.audio, zoom_range=parse_zoom(args.zoom))
