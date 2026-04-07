import os
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
ZIP_PATH = os.path.join(DATA_DIR, "IDMT-SMT-DRUMS-V2.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "IDMT-SMT-DRUMS-V2")

# Zenodo API direct file download URL
ZENODO_URL = "https://zenodo.org/api/records/7544164/files/IDMT-SMT-DRUMS-V2.zip/content"

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if os.path.exists(ZIP_PATH) or os.path.exists(EXTRACT_DIR):
        print("Dataset already partially or fully downloaded.")
    else:
        print(f"Downloading IDMT-SMT-DRUMS dataset to {ZIP_PATH}...")
        response = requests.get(ZENODO_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading")
        
        with open(ZIP_PATH, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print("Download complete.")

def extract_dataset():
    if not os.path.exists(EXTRACT_DIR) and os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH} to {DATA_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset extracted successfully.")
    else:
        print("Extraction already complete, or zip file not found.")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
