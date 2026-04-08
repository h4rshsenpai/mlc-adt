import os
import librosa
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from src.classification import extract_features, WINDOW_SEC, DRUM_CLASSES, MODEL_SAMPLE_RATE

class BaseDataset(ABC):
    """
    Base class for drum transcription datasets.
    Handles common audio loading, onset grouping, and feature extraction.
    """
    
    def __init__(self, dataset_dir, tolerance_sec=0.03):
        self.dataset_dir = dataset_dir
        self.tolerance_sec = tolerance_sec
        self.sr = MODEL_SAMPLE_RATE

    @abstractmethod
    def get_tracks(self):
        """
        Should return a list of dictionaries or objects representing tracks.
        Each track must have enough info to load audio and ground truth.
        """
        pass

    @abstractmethod
    def load_ground_truth(self, track):
        """
        Parses ground truth for a given track.
        Returns a list of (onset_sec, instrument_label) tuples.
        """
        pass

    @abstractmethod
    def get_audio_path(self, track):
        """
        Returns the absolute path to the audio file for a given track.
        """
        pass

    def group_onsets(self, onsets):
        """
        Groups simultaneous hits within tolerance_sec.
        onsets: list of (onset_sec, instrument_label)
        Returns: list of (onset_sec, set(instrument_labels))
        """
        if not onsets:
            return []
            
        onsets.sort(key=lambda x: x[0])
        grouped = []
        current_group_time = onsets[0][0]
        current_group_instruments = {onsets[0][1]}
        
        for onset_sec, instrument in onsets[1:]:
            if onset_sec - current_group_time <= self.tolerance_sec:
                current_group_instruments.add(instrument)
            else:
                grouped.append((current_group_time, current_group_instruments))
                current_group_time = onset_sec
                current_group_instruments = {instrument}
        grouped.append((current_group_time, current_group_instruments))
        return grouped

    def build(self, max_tracks=None, desc="Processing dataset"):
        """
        Iterates through tracks, loads audio, extracts features for each ground truth onset.
        """
        X, y = [], []
        tracks = self.get_tracks()
        
        if max_tracks:
            tracks = tracks[:max_tracks]
            
        for track in tqdm(tracks, desc=desc):
            audio_path = self.get_audio_path(track)
            if not os.path.exists(audio_path):
                continue
                
            onsets = self.load_ground_truth(track)
            if not onsets:
                continue
                
            # Group onsets
            grouped_onsets = self.group_onsets(onsets)
            
            # Load audio
            try:
                audio_data, _ = librosa.load(audio_path, sr=self.sr)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue
                
            prev_onset_sec = None
            for onset_sec, instruments in grouped_onsets:
                start_sample = int(onset_sec * self.sr)
                end_sample = start_sample + int(WINDOW_SEC * self.sr)
                
                if start_sample >= len(audio_data):
                    continue
                    
                y_slice = audio_data[start_sample:end_sample]
                if len(y_slice) == 0:
                    continue
                    
                onset_gap_sec = (onset_sec - prev_onset_sec) if prev_onset_sec is not None else 1.0
                
                features = extract_features(y_slice, self.sr, onset_gap_sec=onset_gap_sec)
                # Map instruments to multi-hot vector based on DRUM_CLASSES
                multi_hot = [1 if cls in instruments else 0 for cls in DRUM_CLASSES]
                
                X.append(features)
                y.append(multi_hot)
                prev_onset_sec = onset_sec
                
        return np.array(X), np.array(y)
