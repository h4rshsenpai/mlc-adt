"""
Refactored evaluation script for PyTorch CNN model.
Supports evaluating a saved model against one or more datasets.
"""
import os
import sys
import torch
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.classification import DRUM_CLASSES, DrumClassifierCNN
from src.scripts.train_classifier import (
    get_dataset, 
    DrumDataset, 
    collate_batch,
    evaluate
)
from torch.utils.data import DataLoader

DATA_DIR = os.path.join(project_root, "data")
MODEL_PATH = os.path.join(DATA_DIR, "drum_classifier.pt")


def evaluate_on_dataset(clf, device, name, data_dir, use_holdout=False, max_tracks=None):
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

        # Create dataset and dataloader
        eval_dataset = DrumDataset(X_eval, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, collate_fn=collate_batch)
        
        # Evaluate
        criterion = torch.nn.BCEWithLogitsLoss()
        val_loss, val_hl, val_acc = evaluate(clf, eval_loader, criterion, device)
        
        # Convert predictions to binary for detailed report
        clf.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                specs = batch["spec"].to(device)
                gap_secs = batch["gap_sec"].to(device)
                labels = batch["label"].to(device)
                predictions = clf(specs, gap_secs)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions_binary = (all_predictions > 0.5).astype(int)
        
        print("\nPER-CLASS REPORT:")
        print(classification_report(all_labels, all_predictions_binary, target_names=DRUM_CLASSES, zero_division=0))

        metrics = {
            "dataset": name,
            "samples": len(X_eval),
            "loss": val_loss,
            "exact_match": val_acc,
            "hamming_loss": val_hl
        }
        
        print("-" * 30)
        print(f"  Loss                 : {val_loss:.4f}")
        print(f"  Exact-Match Accuracy : {val_acc*100:.2f}%")
        print(f"  Hamming Loss         : {val_hl:.4f}")
        
        return metrics
    except Exception as e:
        print(f"Error evaluating on {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drum classifier on benchmark datasets.")
    parser.add_argument("--datasets", type=str, default="gmd", help="Comma-separated list of datasets to test")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the saved model (.pt)")
    parser.add_argument("--holdout", action="store_true", help="Use 20%% holdout split (use if testing on training dataset)")
    parser.add_argument("--max-tracks", type=int, default=None, help="Limit tracks for quick testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at {args.model}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {args.model}...")
    device = args.device
    model = DrumClassifierCNN(n_mels=128, n_classes=len(DRUM_CLASSES))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
        
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    
    results = []
    for name in dataset_list:
        res = evaluate_on_dataset(model, device, name, DATA_DIR, use_holdout=args.holdout, max_tracks=args.max_tracks)
        if res:
            results.append(res)
            
    if len(results) > 1:
        print(f"\n{'='*20} SUMMARY {'='*20}")
        print(f"{'Dataset':<15} | {'Samples':<8} | {'Loss':<8} | {'Exact Match':<12} | {'Hamming Loss':<12}")
        print("-" * 65)
        for r in results:
            print(f"{r['dataset']:<15} | {r['samples']:<8} | {r['loss']:>7.4f} | {r['exact_match']*100:>10.2f}% | {r['hamming_loss']:>12.4f}")
