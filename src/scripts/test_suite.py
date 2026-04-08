import os
import sys
import torch
import xml.etree.ElementTree as ET
import numpy as np
from tabulate import tabulate

# Add src to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.onset import detect_onsets
from src.classification import classify_hits, DRUM_CLASSES, DrumClassifierCNN

MODEL_PATH = os.path.join(project_root, "data", "drum_classifier.pt")
XML_DIR = os.path.join(project_root, "data", "annotation_xml")
TOLERANCE_SEC = 0.030 # 30ms

def load_ground_truth(xml_path):
    if xml_path is None or not os.path.exists(xml_path):
        return None
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []
    
    # Gather all events
    all_events = []
    for event in root.findall('.//event'):
        onset_sec = float(event.find('onsetSec').text)
        instrument = event.find('instrument').text
        if instrument in DRUM_CLASSES:
            all_events.append((onset_sec, instrument))
    
    all_events.sort(key=lambda x: x[0])
    
    if not all_events:
        return []
        
    # Group simultaneous hits (copying logic from train_classifier)
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
    return grouped_events

def evaluate_track(audio_path, xml_path, model, device):
    if not os.path.exists(audio_path):
        return {"Track": os.path.basename(audio_path), "Status": "Skip (File Not Found)"}

    try:
        # Run Pipeline
        y, sr, onset_samples, onset_times, bpm = detect_onsets(audio_path)
        predictions = classify_hits(model, y, sr, onset_samples, device=device)
        
        status_info = {
            "Track": os.path.basename(audio_path),
            "Status": "OK",
            "BPM": f"{bpm:.1f}",
            "Detected": len(onset_times)
        }
        
        gt = load_ground_truth(xml_path)
        if gt is not None:
            # Onset matching
            tp = 0
            unmatched_pred = list(range(len(onset_times)))
            unmatched_gt = list(range(len(gt)))
            
            matched_pairs = [] # (pred_idx, gt_idx)
            
            for p_idx, p_time in enumerate(onset_times):
                best_gt_idx = -1
                best_dist = TOLERANCE_SEC + 1e-6
                
                for g_idx in unmatched_gt:
                    dist = abs(p_time - gt[g_idx][0])
                    if dist <= TOLERANCE_SEC and dist < best_dist:
                        best_dist = dist
                        best_gt_idx = g_idx
                
                if best_gt_idx != -1:
                    tp += 1
                    unmatched_gt.remove(best_gt_idx)
                    matched_pairs.append((p_idx, best_gt_idx))
            
            fp = len(onset_times) - tp
            fn = len(gt) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Classification matching (only for TP onsets)
            class_correct = 0
            for p_idx, g_idx in matched_pairs:
                if set(predictions[p_idx]) == gt[g_idx][1]:
                    class_correct += 1
            
            class_acc = class_correct / tp if tp > 0 else 0
            
            status_info["Onset F1"] = f"{f1:.2f}"
            status_info["Class Acc"] = f"{class_acc:.2f}"
            status_info["GT Count"] = len(gt)
        else:
            status_info["Onset F1"] = "N/A"
            status_info["Class Acc"] = "N/A"
            status_info["GT Count"] = "N/A"
            
        return status_info
        
    except Exception as e:
        return {"Track": os.path.basename(audio_path), "Status": f"Error: {str(e)}"}

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = DrumClassifierCNN(n_mels=128, n_classes=len(DRUM_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    test_cases = [
        # (audio_rel_path, xml_rel_name)
        ("RealDrum01_00#MIX.wav", "RealDrum01_00#MIX.xml"),
        ("data/audio/RealDrum01_11#MIX.wav", "RealDrum01_11#MIX.xml"),
        ("drums.wav", None),
        ("separated/htdemucs/acdc/drums.wav", None)
    ]
    
    # Try alternate paths if not found in current directory
    final_cases = []
    for audio_path, xml_name in test_cases:
        abs_audio = os.path.join(project_root, audio_path)
        if not os.path.exists(abs_audio):
             # Try data/audio/
             alt_audio = os.path.join(project_root, "data", "audio", os.path.basename(audio_path))
             if os.path.exists(alt_audio):
                 abs_audio = alt_audio
        
        abs_xml = os.path.join(XML_DIR, xml_name) if xml_name else None
        final_cases.append((abs_audio, abs_xml))

    results = []
    for audio, xml in final_cases:
        print(f"Evaluating {os.path.basename(audio)}...")
        res = evaluate_track(audio, xml, model, device)
        results.append(res)
    
    print("\n" + "="*80)
    print("ADT TEST SUITE SUMMARY")
    print("="*80)
    print(tabulate(results, headers="keys", tablefmt="grid"))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
