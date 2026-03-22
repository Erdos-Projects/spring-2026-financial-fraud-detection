#!/usr/bin/env python3
"""
Download the Elliptic++ dataset from Google Drive and save it to data/raw/elliptic++dataset/
Google Drive: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
"""

import os
import shutil
import sys

try:
    import gdown
except ImportError:
    print("[ERROR] 'gdown' is not installed. Install it with: pip install gdown")
    sys.exit(1)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TARGET_DIR = os.path.join(RAW_DATA_DIR, "elliptic++dataset")
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"


def download_dataset():
    """Download the Elliptic++ dataset from Google Drive."""

    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # If target directory already exists, ask user
    if os.path.exists(TARGET_DIR):
        print(f"[INFO] Target directory already exists: {TARGET_DIR}")
        response = input("Do you want to re-download? (y/n): ").strip().lower()
        if response != "y":
            print("[INFO] Skipping download.")
            return
        shutil.rmtree(TARGET_DIR)
        print("[INFO] Removed existing directory.")

    print(f"[INFO] Downloading dataset from Google Drive...")
    print(f"[INFO] URL: {GDRIVE_FOLDER_URL}")
    print("[INFO] This may take a while depending on the dataset size...")

    try:
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=TARGET_DIR,
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

    print(f"\n[INFO] Dataset saved to: {TARGET_DIR}")

    # List downloaded files
    print("\n[INFO] Downloaded files:")
    for root, dirs, files in os.walk(TARGET_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        level = root.replace(TARGET_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"{sub_indent}{file} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_dataset()