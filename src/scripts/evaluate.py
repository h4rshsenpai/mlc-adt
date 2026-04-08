"""
Refactored evaluation script.
Supports evaluating a saved model against one or more datasets.
"""
import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.classification import DRUM_CLASSES
from src.scripts.train_classifier import get_dataset

DATA_DIR = os.path.join(project_root, "data")
MODEL_PATH = os.path.join(DATA_DIR, "drum_classifier.pkl")

def evaluate_on_dataset(clf, name, data_dir, use_holdout=False, max_tracks=None):
    print(f"\n{'='*20} Dataset: {name} {'='*20}")
    try:
        ds = get_dataset(name, data_dir)
        X, y = ds.build(max_tracks=max_tracks, desc=f"Loading {name} for evaluation")
        
        if len(X) == 0:
            print(f"No data found for {name}!")
            return None

        if use_holdout:
            print("Using 20% hold-out split (random_state=42)")
            _, X_eval, _, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            print(f"Evaluating on full dataset ({len(X)} samples)")
            X_eval, y_eval = X, y

        y_pred = clf.predict(X_eval)

        print("\nPER-CLASS REPORT:")
        print(classification_report(y_eval, y_pred, target_names=DRUM_CLASSES, zero_division=0))

        exact_match = accuracy_score(y_eval, y_pred)
        hl = hamming_loss(y_eval, y_pred)
        
        metrics = {
            "dataset": name,
            "samples": len(X_eval),
            "exact_match": exact_match,
            "hamming_loss": hl
        }
        
        print("-" * 30)
        print(f"  Exact-Match Accuracy : {exact_match*100:.2f}%")
        print(f"  Hamming Loss         : {hl:.4f}")
        
        return metrics
    except Exception as e:
        print(f"Error evaluating on {name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drum classifier on benchmark datasets.")
    parser.add_argument("--datasets", type=str, default="gmd", help="Comma-separated list of datasets to test")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the saved model (.pkl)")
    parser.add_argument("--holdout", action="store_true", help="Use 20%% holdout split (use if testing on training dataset)")
    parser.add_argument("--max-tracks", type=int, default=None, help="Limit tracks for quick testing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at {args.model}")
        sys.exit(1)
        
    print(f"Loading model from {args.model}...")
    with open(args.model, "rb") as f:
        clf = pickle.load(f)
        
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    
    results = []
    for name in dataset_list:
        res = evaluate_on_dataset(clf, name, DATA_DIR, use_holdout=args.holdout, max_tracks=args.max_tracks)
        if res:
            results.append(res)
            
    if len(results) > 1:
        print(f"\n{'='*20} SUMMARY {'='*20}")
        print(f"{'Dataset':<15} | {'Samples':<8} | {'Exact Match':<12} | {'Hamming Loss':<12}")
        print("-" * 55)
        for r in results:
            print(f"{r['dataset']:<15} | {r['samples']:<8} | {r['exact_match']*100:>10.2f}% | {r['hamming_loss']:>12.4f}")
