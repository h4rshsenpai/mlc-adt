import os
import xml.etree.ElementTree as ET
from .base import BaseDataset

class IDMTDataset(BaseDataset):
    """
    Implementation for IDMT-SMT-Drums dataset.
    Expects structure:
    dataset_dir/
      audio/
      annotation_xml/
    """
    def __init__(self, dataset_dir, tolerance_sec=0.03):
        super().__init__(dataset_dir, tolerance_sec)
        self.audio_dir = os.path.join(dataset_dir, "audio")
        self.xml_dir = os.path.join(dataset_dir, "annotation_xml")

    def get_tracks(self):
        if not os.path.exists(self.xml_dir):
            print(f"ERROR: {self.xml_dir} not found!")
            return []
            
        # Each XML file corresponds to a track
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith(".xml")]
        return sorted(xml_files)

    def get_audio_path(self, track):
        # track is the filename of the XML
        audio_filename = track.replace(".xml", ".wav")
        return os.path.join(self.audio_dir, audio_filename)

    def load_ground_truth(self, track):
        xml_path = os.path.join(self.xml_dir, track)
        if not os.path.exists(xml_path):
            return []
            
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            events = []
            
            # Find all <event> tags in <transcription>
            for event in root.findall(".//event"):
                onset_sec = float(event.find("onsetSec").text)
                instrument = event.find("instrument").text
                # Unified mapping happens here
                # IDMT uses HH, KD, SD which match our DRUM_CLASSES
                if instrument in ['HH', 'KD', 'SD']:
                    events.append((onset_sec, instrument))
            
            return events
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return []
