import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

# Add `src` to sys.path so we can import our modules locally
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.classification import DRUM_CLASSES, DrumClassifierCNN
from src.datasets.gmd import GMDDataset
from src.datasets.idmt import IDMTDataset

# Paths
DATA_DIR = os.path.join(project_root, "data")
MODEL_PATH = os.path.join(DATA_DIR, "drum_classifier.pt")


def get_dataset(name, data_dir):
    """Factory function to load a dataset by name."""
    if name.lower() == "gmd":
        return GMDDataset(os.path.join(data_dir, "groove"))
    elif name.lower() == "idmt":
        return IDMTDataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def build_combined_dataset(dataset_names, data_dir, max_tracks_per_dataset=None):
    """
    Builds a combined dataset from multiple sources.
    
    Returns:
        X (list[dict]): List of dicts with keys 'spec' (np.ndarray) and 'gap_sec' (float)
        y (np.ndarray): Multi-hot labels, shape (n_samples, n_classes)
    """
    X_all, y_all = [], []
    
    for name in dataset_names:
        print(f"Loading dataset: {name}...")
        try:
            ds = get_dataset(name, data_dir)
            X, y = ds.build(max_tracks=max_tracks_per_dataset, desc=f"Building {name}")
            if len(X) > 0:
                X_all.extend(X)  # X is a list of dicts
                y_all.append(y)
                print(f"  Loaded {len(X)} samples from {name}")
            else:
                print(f"  No samples found in {name}")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    
    if not X_all:
        return [], np.array([])
    
    return X_all, np.concatenate(y_all, axis=0)


class DrumDataset(Dataset):
    """
    PyTorch Dataset for drum transcription.
    
    X: list of dicts with keys 'spec' (np.ndarray shape (n_mels, n_frames, 1)) 
       and 'gap_sec' (float)
    y: np.ndarray of shape (n_samples, n_classes) with multi-hot labels
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y), "Length mismatch between X and y"
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        spec = self.X[idx]["spec"]  # (n_mels, n_frames, 1)
        gap_sec = self.X[idx]["gap_sec"]  # float
        label = self.y[idx]  # (n_classes,)
        
        return {
            "spec": torch.from_numpy(spec).float(),
            "gap_sec": torch.tensor(gap_sec, dtype=torch.float32),
            "label": torch.from_numpy(label).float()
        }


def collate_batch(batch):
    """
    Custom collate function to handle variable-length spectrograms.
    
    Pads all spectrograms to the max length in the batch, then rearranges to
    channel-first format for PyTorch Conv2D.
    """
    specs = [item["spec"] for item in batch]
    gap_secs = torch.stack([item["gap_sec"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    # Find max frames in batch
    max_frames = max(s.shape[1] for s in specs)
    
    # Pad all specs to max length
    specs_padded = []
    for s in specs:
        n_frames = s.shape[1]
        if n_frames < max_frames:
            # Pad along the frame (time) dimension: (n_mels, n_frames, 1) -> (n_mels, max_frames, 1)
            pad_amount = max_frames - n_frames
            s = F.pad(s, (0, 0, 0, pad_amount))  # pad=(left, right, top, bottom, ...) for last dims first
        specs_padded.append(s)
    
    # Stack: (batch, n_mels, max_frames, 1)
    specs_batch = torch.stack(specs_padded, dim=0)
    
    # Rearrange to channel-first format: (batch, 1, n_mels, max_frames)
    specs_batch = specs_batch.permute(0, 3, 1, 2)
    
    return {
        "spec": specs_batch,
        "gap_sec": gap_secs,
        "label": labels
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        specs = batch["spec"].to(device)
        gap_secs = batch["gap_sec"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        predictions = model(specs, gap_secs)
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, eval_loader, criterion, device):
    """Evaluate on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            specs = batch["spec"].to(device)
            gap_secs = batch["gap_sec"].to(device)
            labels = batch["label"].to(device)
            
            predictions = model(specs, gap_secs)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Convert to numpy for metrics
            pred_np = predictions.cpu().numpy()
            label_np = labels.cpu().numpy()
            
            all_predictions.append(pred_np)
            all_labels.append(label_np)
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Convert to binary (threshold at 0.5)
    all_predictions_binary = (all_predictions > 0.5).astype(int)
    
    # Compute metrics
    hl = hamming_loss(all_labels, all_predictions_binary)
    accuracy = accuracy_score(all_labels, all_predictions_binary)
    
    return total_loss / num_batches if num_batches > 0 else 0.0, hl, accuracy


def train_and_evaluate(X, y_labels, model_path=MODEL_PATH, device='cpu', num_epochs=100, 
                      batch_size=128, learning_rate=1e-3, early_stopping_patience=10):
    """
    Train the CNN model.
    
    Args:
        X: list of dicts with 'spec' and 'gap_sec'
        y_labels: np.ndarray of multi-hot labels
        model_path: path to save the trained model
        device: 'cpu' or 'cuda'
        num_epochs: maximum number of epochs
        batch_size: batch size for training
        learning_rate: learning rate for Adam optimizer
        early_stopping_patience: stop if validation loss doesn't improve for N epochs
    """
    if len(X) == 0:
        print("No training data generated. Check file paths and data availability.")
        return
    
    print(f"\nExtracted {len(X)} instances. Splitting (80/20 train/val)...")
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Create datasets and dataloaders
    train_dataset = DrumDataset(X_train, y_train)
    val_dataset = DrumDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    
    # Initialize model, loss, optimizer
    print(f"\nInitializing CNN model on device: {device}")
    model = DrumClassifierCNN(n_mels=128, n_classes=len(DRUM_CLASSES)).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining for max {num_epochs} epochs with early stopping patience={early_stopping_patience}...\n")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_hl, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Hamming Loss: {val_hl:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved! Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered (patience={early_stopping_patience})")
                break
    
    # Load best model and evaluate on full validation set
    print(f"\nLoading best model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    final_val_loss, final_hl, final_acc = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {final_val_loss:.4f}")
    print(f"  Hamming Loss: {final_hl:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    
    print(f"\nModel training complete. Saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drum classifier CNN on one or more datasets.")
    parser.add_argument("--datasets", type=str, default="gmd", 
                       help="Comma-separated list of datasets (gmd, idmt)")
    parser.add_argument("--max-tracks", type=int, default=None, 
                       help="Max tracks per dataset for quick testing")
    parser.add_argument("--output", type=str, default=MODEL_PATH, 
                       help="Path to save the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to train on (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=128, 
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    
    print(f"Device: {args.device}")
    print(f"Datasets: {', '.join(dataset_list)}")
    
    X, y = build_combined_dataset(dataset_list, DATA_DIR, max_tracks_per_dataset=args.max_tracks)
    train_and_evaluate(
        X, y, 
        model_path=args.output,
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
