import os
import requests
import zipfile
import argparse
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# Dataset configurations
DATASETS = {
    "gmd": {
        "url": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip",
        "zip_name": "groove-v1.0.0.zip",
        "extract_to": "groove",
        "check_path": "groove/info.csv"
    },
    "idmt": {
        "url": "https://www.idmt.fraunhofer.de/content/dam/idmt/en/documents/Datasets/IDMT-SMT-DRUMS-V2.zip",
        "zip_name": "IDMT-SMT-DRUMS-V2.zip",
        "extract_to": ".", # Extracts to audio/, annotation_xml/, etc. directly
        "check_path": "annotation_xml"
    }
}

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Download")
    
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    print("Download complete.")

def extract_file(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def fetch_dataset(name):
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        return

    config = DATASETS[name]
    zip_path = os.path.join(DATA_DIR, config["zip_name"])
    check_path = os.path.join(DATA_DIR, config["check_path"])

    if os.path.exists(check_path):
        print(f"Dataset '{name}' already exists at {check_path}.")
        return

    if not os.path.exists(zip_path):
        download_file(config["url"], zip_path)
    
    extract_dir = os.path.join(DATA_DIR, config["extract_to"]) if config["extract_to"] != "." else DATA_DIR
    extract_file(zip_path, extract_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch benchmark datasets.")
    parser.add_argument("dataset", type=str, nargs="?", default="gmd", help="Dataset name (gmd, idmt)")
    parser.add_argument("--all", action="store_true", help="Fetch all supported datasets")
    
    args = parser.parse_args()
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if args.all:
        for ds in DATASETS:
            fetch_dataset(ds)
    else:
        fetch_dataset(args.dataset)
