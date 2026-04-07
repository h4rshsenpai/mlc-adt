"""
Standalone evaluation script.
Rebuilds the dataset, applies the same train/test split (random_state=42),
then loads the saved model and evaluates only on the held-out test portion.
Does NOT retrain.
"""
import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.scripts.train_classifier import build_dataset
from src.classification import DRUM_CLASSES

MODEL_PATH = os.path.join(project_root, "data", "drum_classifier.pkl")

def evaluate():
    print("Rebuilding dataset (no training)...")
    X, y = build_dataset()

    if len(X) == 0:
        print("No data found!")
        return

    print(f"Total samples: {len(X)}")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Test samples (held-out 20%): {len(X_test)}\n")

    print(f"Loading saved model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    y_pred = clf.predict(X_test)

    print("=" * 50)
    print("PER-CLASS REPORT (multi-label, per column)")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=DRUM_CLASSES, zero_division=0))

    print("=" * 50)
    print("AGGREGATE METRICS")
    print("=" * 50)

    # Exact match accuracy: fraction of samples where ALL labels match perfectly
    exact_match = accuracy_score(y_test, y_pred)
    print(f"  Exact-Match Accuracy : {exact_match*100:.2f}%")

    # Hamming loss: fraction of individual label predictions that are wrong
    hl = hamming_loss(y_test, y_pred)
    print(f"  Hamming Loss         : {hl:.4f}  (lower is better, 0 = perfect)")

if __name__ == "__main__":
    evaluate()
