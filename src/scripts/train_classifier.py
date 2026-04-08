import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add `src` to sys.path so we can import our modules locally
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.classification import DRUM_CLASSES
from src.datasets.gmd import GMDDataset
from src.datasets.idmt import IDMTDataset

# Paths
DATA_DIR = os.path.join(project_root, "data")
MODEL_PATH = os.path.join(DATA_DIR, "drum_classifier.pkl")

def get_dataset(name, data_dir):
    if name.lower() == "gmd":
        return GMDDataset(os.path.join(data_dir, "groove"))
    elif name.lower() == "idmt":
        return IDMTDataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def build_combined_dataset(dataset_names, data_dir, max_tracks_per_dataset=None):
    """
    Builds a combined dataset from multiple sources.
    """
    X_all, y_all = [], []
    
    for name in dataset_names:
        print(f"Loading dataset: {name}...")
        try:
            ds = get_dataset(name, data_dir)
            X, y = ds.build(max_tracks=max_tracks_per_dataset, desc=f"Building {name}")
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                print(f"  Loaded {len(X)} samples from {name}")
            else:
                print(f"  No samples found in {name}")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            
    if not X_all:
        return np.array([]), np.array([])
        
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

def train_and_evaluate(X, y_labels, model_path=MODEL_PATH):
    if len(X) == 0:
        print("No training data generated. Check file paths and data availability.")
        return
        
    print(f"\nExtracted {len(X)} instances. Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.15, random_state=42)
    
    print("Fitting Random Forest (n_jobs=-1 for speed)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    print("\nMetrics on internal test split:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=DRUM_CLASSES, zero_division=0))
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drum classifier on one or more datasets.")
    parser.add_argument("--datasets", type=str, default="gmd", help="Comma-separated list of datasets (gmd, idmt)")
    parser.add_argument("--max-tracks", type=int, default=None, help="Max tracks per dataset for quick testing")
    parser.add_argument("--output", type=str, default=MODEL_PATH, help="Path to save the model")
    
    args = parser.parse_args()
    
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    
    X, y = build_combined_dataset(dataset_list, DATA_DIR, max_tracks_per_dataset=args.max_tracks)
    train_and_evaluate(X, y, model_path=args.output)
