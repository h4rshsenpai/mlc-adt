import os
import sys
import xml.etree.ElementTree as ET
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

DATA_DIR = os.path.join(project_root, "data")
XML_DIR = os.path.join(DATA_DIR, "annotation_xml")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MODEL_PATH = os.path.join(DATA_DIR, "drum_classifier.pkl")

# Tolerance for grouping simultaneous hits (in seconds)
TOLERANCE_SEC = 0.03

def build_dataset():
    """
    Parses all XML files and extracts features from the corresponding audio.
    Aggregates simultaneous hits into multi-hot binary vectors.
    """
    X, y_labels = [], []
    
    if not os.path.exists(XML_DIR):
        print("ERROR: XML directory not found. Did you run fetch_dataset.py?")
        return X, y_labels
        
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"Discovered {len(xml_files)} annotation files. Parsing and extracting features...")
    
    for xml_file in tqdm(xml_files, desc="Processing Tracks"):
        base_name = xml_file.replace('.xml', '.wav')
        audio_path = os.path.join(AUDIO_DIR, base_name)
        
        if not os.path.exists(audio_path):
            continue
            
        tree = ET.parse(os.path.join(XML_DIR, xml_file))
        root = tree.getroot()
        
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # 1. Gather and sort all events by timestamp
        all_events = []
        for event in root.findall('.//event'):
            onset_sec = float(event.find('onsetSec').text)
            instrument = event.find('instrument').text
            if instrument in DRUM_CLASSES:
                all_events.append((onset_sec, instrument))
                
        all_events.sort(key=lambda x: x[0])
        
        # 2. Group simultaneous hits
        if not all_events:
            continue
            
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
        
        # 3. Extract features for each group, including inter-onset gap
        prev_onset_sec = None
        for onset_sec, instruments in grouped_events:
            start_sample = int(onset_sec * sr)
            end_sample = start_sample + int(WINDOW_SEC * sr)
            y_slice = audio_data[start_sample:end_sample]

            if len(y_slice) > 0:
                # Inter-onset gap: time since the previous event. 1.0s default for first hit.
                onset_gap_sec = (onset_sec - prev_onset_sec) if prev_onset_sec is not None else 1.0
                features = extract_features(y_slice, sr, onset_gap_sec=onset_gap_sec)

                # Multi-hot encoding for DRUM_CLASSES list order
                multi_hot = [1 if cls in instruments else 0 for cls in DRUM_CLASSES]

                X.append(features)
                y_labels.append(multi_hot)

            prev_onset_sec = onset_sec

    return np.array(X), np.array(y_labels)


def train_and_evaluate(X, y_labels):
    if len(X) == 0:
        print("No training data found!")
        return
        
    print(f"\nExtracted {len(X)} unique multi-label drum events! Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)
    
    print("Training Multi-Label Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Scikit-learn Random Forest handles 2D arrays natively for multi-label classification
    clf.fit(X_train, y_train)
    
    print("\nEvaluatiing on unseen test split:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=DRUM_CLASSES, zero_division=0))
    
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print("Done!")

if __name__ == "__main__":
    X, y = build_dataset()
    train_and_evaluate(X, y)
