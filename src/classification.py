import os
import librosa
import numpy as np
from scipy.signal import butter, filtfilt

WINDOW_SEC = 0.100  # 100ms window after onset
DRUM_CLASSES = ['KD', 'SD', 'HH']  # Standard class ordering for multi-hot vectors
HIGHPASS_CUTOFF_HZ = 250  # Suppress kick bleed-over (kick energy mostly lives below 200Hz)

def _highpass_filter(y, sr, cutoff=HIGHPASS_CUTOFF_HZ):
    """
    Apply a 2nd-order Butterworth high-pass filter to suppress low-frequency bleed
    from preceding kicks into the snare/hihat feature window.
    """
    nyquist = sr / 2.0
    normed = cutoff / nyquist
    b, a = butter(2, normed, btype='high')
    return filtfilt(b, a, y)

def extract_features(y, sr, onset_gap_sec=1.0):
    """
    Extracts acoustic features from an audio slice.

    Features:
        - 20 MFCCs on high-pass filtered audio (removes preceding kick bleed)
        - Spectral Centroid: brightness (hi-hats = high, kicks = low)
        - Spectral Rolloff: upper frequency cutoff (helps separate SD+HH from pure SD)
        - Spectral Bandwidth: width of frequency band
        - ZCR: noisiness proxy (snare rattles = high ZCR)
        - onset_gap_sec: time since previous onset — helps model learn that hits
          very close after a KD are likely SD/HH rather than another KD
    """
    if len(y) < 2048:
        y = np.pad(y, (0, max(0, 2048 - len(y))))

    # Apply high-pass filter to suppress preceding kick's low-end decay bleed
    y_filtered = _highpass_filter(y, sr)

    mfccs = np.mean(librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=20).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_filtered, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_filtered, sr=sr, roll_percent=0.85))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_filtered))

    return np.hstack((mfccs, centroid, rolloff, bandwidth, zcr, onset_gap_sec))

def classify_hits(model, y, sr, onset_samples):
    """
    Slices the audio at each onset sample and uses the provided Multi-Label model to classify it.
    Computes inter-onset gaps automatically from onset_samples.
    Returns a list of lists of active label strings (e.g., [['KD', 'HH'], ['SD'], ...]).
    """
    features_list = []
    valid_indices = []

    for idx, sample in enumerate(onset_samples):
        end_sample = sample + int(WINDOW_SEC * sr)
        y_slice = y[sample:end_sample]

        if len(y_slice) == 0:
            continue

        # Inter-onset gap: time from previous onset. 1.0s default for the first hit.
        if idx == 0:
            onset_gap_sec = 1.0
        else:
            onset_gap_sec = float(onset_samples[idx] - onset_samples[idx - 1]) / sr

        features = extract_features(y_slice, sr, onset_gap_sec=onset_gap_sec)
        features_list.append(features)
        valid_indices.append(idx)

    if not features_list:
        return [[] for _ in onset_samples]

    X = np.array(features_list)
    # Model predictions: 2D binary array of shape (n_samples, n_classes)
    predictions = model.predict(X)

    final_output = [[] for _ in onset_samples]
    for i, valid_idx in enumerate(valid_indices):
        active_classes = [DRUM_CLASSES[j] for j, is_active in enumerate(predictions[i]) if is_active]
        final_output[valid_idx] = active_classes

    return final_output
