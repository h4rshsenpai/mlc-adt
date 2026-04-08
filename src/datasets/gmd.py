import os
import csv
import mido
from .base import BaseDataset

# MIDI Note to Class Mapping (General MIDI standard) - same as in train_classifier.py
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

class GMDDataset(BaseDataset):
    """
    Implementation for the Groove MIDI Dataset (GMD).
    """
    def __init__(self, dataset_dir, splits=None, tolerance_sec=0.03):
        super().__init__(dataset_dir, tolerance_sec)
        self.info_csv = os.path.join(dataset_dir, "info.csv")
        self.splits = splits

    def get_tracks(self):
        if not os.path.exists(self.info_csv):
            print(f"ERROR: {self.info_csv} not found!")
            return []
            
        tracks = []
        with open(self.info_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.splits and row.get("split") not in self.splits:
                    continue
                if row.get('audio_filename'):
                    tracks.append(row)
        return tracks

    def get_audio_path(self, track):
        return os.path.join(self.dataset_dir, track['audio_filename'])

    def load_ground_truth(self, track):
        midi_rel_path = track['midi_filename']
        midi_path = os.path.join(self.dataset_dir, midi_rel_path)
        
        if not os.path.exists(midi_path):
            return []
            
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
