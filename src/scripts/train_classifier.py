import os
import sys
import csv
import mido
import librosa
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add `src` to sys.path so we can import our modules locally
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.classification import extract_features, WINDOW_SEC, DRUM_CLASSES

# Paths
DATA_DIR = os.path.join(project_root, "data")
GMD_DIR = os.path.join(DATA_DIR, "groove")
INFO_CSV = os.path.join(GMD_DIR, "info.csv")
MODEL_PATH = os.path.join(DATA_DIR, "drum_classifier.pkl")

# MIDI Note to Class Mapping (General MIDI standard)
MIDI_MAPPING = {
    # Kick
    36: 'KD',
    # Snare
    38: 'SD', 40: 'SD', 
    # Hi-Hat
    42: 'HH', 44: 'HH', 46: 'HH',
    # Toms
    41: 'TT', 43: 'TT', 45: 'TT', 47: 'TT', 48: 'TT', 50: 'TT',
    # Cymbals
    49: 'CY', 51: 'CY', 52: 'CY', 53: 'CY', 55: 'CY', 57: 'CY', 59: 'CY'
}

# Tolerance for grouping simultaneous hits (in seconds)
TOLERANCE_SEC = 0.03


def _parse_int_env(name):
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        print(f"Warning: ignoring invalid {name}={raw!r}")
        return None


def _parse_str_set_env(name):
    raw = os.getenv(name)
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None

def parse_midi_onsets(midi_path):
    """
    Parses a MIDI file and extracts the absolute time (seconds) of each drum hit.
    """
    try:
        mid = mido.MidiFile(midi_path)
        events = []
        current_time = 0
        
        for msg in mid:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                pitch = msg.note
                if pitch in MIDI_MAPPING:
                    events.append((current_time, MIDI_MAPPING[pitch]))
        
        return events
    except Exception as e:
        print(f"Error parsing MIDI {midi_path}: {e}")
        return []

def build_dataset():
    """
    Iterates through GMD's info.csv, loads WAV audio, parses corresponding MIDI,
    and extracts multi-label features for training.
    """
    X, y_labels = [], []
    max_tracks = _parse_int_env("GMD_MAX_TRACKS")
    allowed_splits = _parse_str_set_env("GMD_SPLITS")
    
    if not os.path.exists(INFO_CSV):
        print(f"ERROR: {INFO_CSV} not found! Did you extract GMD correctly?")
        return X, y_labels
        
    with open(INFO_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if allowed_splits is not None:
        rows = [row for row in rows if row.get("split") in allowed_splits]

    if max_tracks is not None:
        rows = rows[:max_tracks]
        
    print(
        f"Discovered {len(rows)} entries in GMD metadata"
        + (f" (splits={sorted(allowed_splits)})" if allowed_splits else "")
        + (f" (capped at {max_tracks})" if max_tracks else "")
        + ". Filtering and processing files..."
    )
    
    # Optional: limit rows during development if needed, but user said "all of it"
    # rows = rows[:100] 

    for row in tqdm(rows, desc="Processing GMD Sessions"):
        audio_rel_path = row['audio_filename']
        midi_rel_path = row['midi_filename']
        
        if not audio_rel_path:
            continue
            
        audio_path = os.path.join(GMD_DIR, audio_rel_path)
        midi_path = os.path.join(GMD_DIR, midi_rel_path)
        
        if not os.path.exists(audio_path) or not os.path.exists(midi_path):
            continue
            
        # Parse MIDI for ground-truth onsets
        all_events = parse_midi_onsets(midi_path)
        if not all_events:
            continue
        
        # Load audio (downsample to 22.05kHz for speed)
        try:
            audio_data, sr = librosa.load(audio_path, sr=22050)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue
        
        # 1. Group simultaneous hits
        all_events.sort(key=lambda x: x[0])
        grouped_events = []
        current_group_time = all_events[0][0]
        current_group_instruments = {all_events[0][1]}
        
        for onset_sec, instrument in all_events[1:]:
            if onset_sec - current_group_time <= TOLERANCE_SEC:
                current_group_instruments.add(instrument)
            else:
                grouped_events.append((current_group_time, current_group_instruments))
                current_group_time = onset_sec
                current_group_instruments = {instrument}
        grouped_events.append((current_group_time, current_group_instruments))
        
        # 2. Extract features
        prev_onset_sec = None
        for onset_sec, instruments in grouped_events:
            start_sample = int(onset_sec * sr)
            end_sample = start_sample + int(WINDOW_SEC * sr)
            
            if start_sample >= len(audio_data):
                continue
                
            y_slice = audio_data[start_sample:end_sample]
            if len(y_slice) == 0:
                continue
                
            onset_gap_sec = (onset_sec - prev_onset_sec) if prev_onset_sec is not None else 1.0
            
            features = extract_features(y_slice, sr, onset_gap_sec=onset_gap_sec)
            multi_hot = [1 if cls in instruments else 0 for cls in DRUM_CLASSES]
            
            X.append(features)
            y_labels.append(multi_hot)
            prev_onset_sec = onset_sec

    return np.array(X), np.array(y_labels)

def train_and_evaluate(X, y_labels):
    if len(X) == 0:
        print("No training data generated. Check file paths and data availability.")
        return
        
    print(f"\nExtracted {len(X)} instances. Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.15, random_state=42)
    
    print("Fitting Random Forest (n_jobs=-1 for speed)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    print("\nMetrics:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=DRUM_CLASSES, zero_division=0))
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    X, y = build_dataset()
    train_and_evaluate(X, y)
