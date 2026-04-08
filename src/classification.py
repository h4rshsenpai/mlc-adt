import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW_SEC = 0.100  # 100ms window after onset
MODEL_SAMPLE_RATE = 22050  # Keep train/inference feature extraction on the same frequency scale
DRUM_CLASSES = ['KD', 'SD', 'HH', 'TT', 'CY']  # KD (Kick), SD (Snare), HH (Hi-Hat), TT (Toms), CY (Cymbals)

# Mel-Spectrogram hyperparameters
MEL_N_MELS = 128
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512


def extract_mel_spectrogram(y, sr, n_mels=MEL_N_MELS, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, normalize=True):
    """
    Extract a Mel-Spectrogram from an audio slice.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        n_mels (int): Number of mel frequency bins
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        normalize (bool): Whether to z-score normalize the spectrogram
    
    Returns:
        np.ndarray: Mel-spectrogram with shape (n_mels, n_frames, 1)
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    
    # Convert power to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize (z-score)
    if normalize:
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / std
        else:
            mel_spec_db = mel_spec_db - mean
    
    # Add channel dimension: (n_mels, n_frames) -> (n_mels, n_frames, 1)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    
    return mel_spec_db.astype(np.float32)


class DrumClassifierCNN(nn.Module):
    """
    CNN for multi-label drum classification.
    
    Input shape: (batch, 1, 128, n_frames)
    Output shape: (batch, 5)
    
    Architecture:
    - Conv2D(32, 3x3) -> ReLU -> MaxPool(2x2)
    - Conv2D(64, 3x3) -> ReLU -> MaxPool(2x2)
    - AdaptiveAvgPool2d(1, 1) -> Flatten
    - Concatenate with onset_gap_sec feature
    - Dense(128) -> ReLU -> Dropout
    - Dense(5) -> Sigmoid (multi-label)
    """
    
    def __init__(self, n_mels=128, n_classes=5, dropout_rate=0.3):
        super(DrumClassifierCNN, self).__init__()
        self.n_classes = n_classes
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to fixed spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers (after concatenating with onset_gap_sec)
        self.fc1 = nn.Linear(64 + 1, 128)  # 64 from CNN + 1 for onset_gap_sec
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, spec_batch, gap_sec_batch):
        """
        Args:
            spec_batch (torch.Tensor): Shape (batch, 1, 128, n_frames)
            gap_sec_batch (torch.Tensor): Shape (batch,)
        
        Returns:
            torch.Tensor: Shape (batch, 5) with sigmoid activation
        """
        # Convolutional blocks
        x = self.conv1(spec_batch)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 64)
        
        # Concatenate with onset_gap_sec
        gap_sec_expanded = gap_sec_batch.unsqueeze(1)  # (batch, 1)
        x = torch.cat([x, gap_sec_expanded], dim=1)  # (batch, 65)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 5)
        
        # Multi-label sigmoid activation
        x = torch.sigmoid(x)
        
        return x


def classify_hits(model, y, sr, onset_samples, device='cpu'):
    """
    Slices the audio at each onset sample and uses the provided CNN model to classify it.
    Computes inter-onset gaps automatically from onset_samples.
    Returns a list of lists of active label strings (e.g., [['KD', 'HH'], ['SD'], ...]).
    
    Args:
        model (DrumClassifierCNN): Trained CNN model
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        onset_samples (np.ndarray): Sample indices of detected onsets
        device (str): 'cpu' or 'cuda'
    
    Returns:
        list[list[str]]: Multi-label predictions per onset
    """
    specs_list = []
    gap_secs_list = []
    valid_indices = []
    
    for idx, sample in enumerate(onset_samples):
        end_sample = sample + int(WINDOW_SEC * sr)
        y_slice = y[sample:end_sample]
        
        if len(y_slice) == 0:
            continue
        
        # Compute inter-onset gap
        if idx == 0:
            onset_gap_sec = 1.0
        else:
            onset_gap_sec = float(onset_samples[idx] - onset_samples[idx - 1]) / sr
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(y_slice, sr)  # (n_mels, n_frames, 1)
        specs_list.append(mel_spec)
        gap_secs_list.append(onset_gap_sec)
        valid_indices.append(idx)
    
    if not specs_list:
        return [[] for _ in onset_samples]
    
    # Batch processing: pad specs to max length in batch
    max_frames = max(spec.shape[1] for spec in specs_list)
    specs_padded = []
    for spec in specs_list:
        n_frames = spec.shape[1]
        if n_frames < max_frames:
            # Pad with zeros: (n_mels, n_frames, 1) -> (n_mels, max_frames, 1)
            pad_amount = max_frames - n_frames
            spec = np.pad(spec, ((0, 0), (0, pad_amount), (0, 0)), mode='constant', constant_values=0)
        specs_padded.append(spec)
    
    # Stack and rearrange to channel-first format
    specs_batch = np.stack(specs_padded, axis=0)  # (batch, n_mels, max_frames, 1)
    specs_batch = np.transpose(specs_batch, (0, 3, 1, 2))  # (batch, 1, n_mels, max_frames)
    
    # Convert to torch tensors
    specs_tensor = torch.from_numpy(specs_batch).float().to(device)
    gap_secs_tensor = torch.from_numpy(np.array(gap_secs_list)).float().to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(specs_tensor, gap_secs_tensor)  # (batch, 5)
    
    # Convert to numpy and threshold at 0.5 for multi-label
    predictions_np = predictions.cpu().numpy()
    predictions_binary = (predictions_np > 0.5).astype(int)
    
    # Map back to original onset indices
    final_output = [[] for _ in onset_samples]
    for i, valid_idx in enumerate(valid_indices):
        active_classes = [DRUM_CLASSES[j] for j in range(len(DRUM_CLASSES)) if predictions_binary[i, j] == 1]
        final_output[valid_idx] = active_classes
    
    return final_output
