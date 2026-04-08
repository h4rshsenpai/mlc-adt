import os
import tempfile

# Matplotlib/fontconfig want writable cache directories very early during import.
# In sandboxed Codex runs, $HOME and the Homebrew cache paths may be read-only, so
# force both fontconfig and Matplotlib to use a writable temp location.
_cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(tempfile.gettempdir(), "codex-cache")
os.makedirs(_cache_root, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "matplotlib"))

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.classification import MODEL_SAMPLE_RATE

DEFAULT_HOP_LENGTH = 512  # librosa default

def estimate_tempo(y, sr):
    """
    Estimates song tempo (BPM) and derives a Minimum Inter-Onset Interval (MIOI).
    The MIOI is half the duration of a 16th note at the estimated tempo, giving a
    musically-grounded refractory window that adapts to any tempo.

    Returns:
        bpm: Estimated tempo in beats per minute.
        mioi_samples: The minimum inter-onset interval in samples (to pass to librosa).
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
    bpm = float(np.atleast_1d(tempo)[0])

    # Smallest subdivision = 16th note = quarter of one beat
    sixteenth_note_sec = (60.0 / bpm) / 4.0

    # Use half of a 16th note as the refractory window — conservative enough to
    # kill echo/rattle re-triggers but loose enough to allow fast ghost notes
    mioi_sec = sixteenth_note_sec * 0.5
    mioi_samples = int(mioi_sec * sr / DEFAULT_HOP_LENGTH)  # convert to frames for librosa

    return bpm, mioi_samples

def detect_onsets(audio_path, target_sr=MODEL_SAMPLE_RATE):
    """
    Loads an audio file, estimates its tempo, and detects onsets with a
    musically-grounded refractory window.

    Returns:
        y: Raw audio time series.
        sr: Sampling rate.
        onset_samples: Array of sample indices where onsets occur.
        onset_times: Array of time (in seconds) where onsets occur.
        bpm: Estimated tempo.
    """
    print(f"[{os.path.basename(audio_path)}] Loading audio...")
    y, sr = librosa.load(audio_path, sr=target_sr)

    print(f"[{os.path.basename(audio_path)}] Estimating tempo...")
    bpm, mioi_frames = estimate_tempo(y, sr)
    print(f"[{os.path.basename(audio_path)}] Estimated BPM: {bpm:.1f} — refractory window: {(mioi_frames * DEFAULT_HOP_LENGTH / sr * 1000):.1f}ms ({mioi_frames} frames)")

    print(f"[{os.path.basename(audio_path)}] Calculating onset envelopes...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=DEFAULT_HOP_LENGTH)

    print(f"[{os.path.basename(audio_path)}] Detecting onset events...")
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        wait=mioi_frames,
    )

    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=DEFAULT_HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=DEFAULT_HOP_LENGTH)

    print(f"[{os.path.basename(audio_path)}] Detected {len(onset_samples)} hits.")
    return y, sr, onset_samples, onset_times, bpm

def plot_onsets(y, sr, onset_times, output_path, title=None, zoom_sec=None):
    """
    Plots the waveform and visualizes the given onset timestamps using vertical red lines.
    """
    print(f"Plotting waveform and onsets to {output_path}...")
    plt.figure(figsize=(14, 5))

    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.vlines(onset_times, ymin=-1, ymax=1, color='r', alpha=0.9, linestyle='--', label='Onsets')

    if zoom_sec:
        plt.xlim(0, zoom_sec)
        if title:
            title += f" (First {zoom_sec}s)"

    plt.title(title if title else "Waveform & Detected Onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
