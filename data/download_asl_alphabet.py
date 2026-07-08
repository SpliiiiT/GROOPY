"""Download and unpack the ASL Alphabet dataset from Kaggle.

Requires a Kaggle API token:
  1. kaggle.com -> Account -> Create New API Token  (downloads kaggle.json)
  2. Put kaggle.json at ~/.kaggle/kaggle.json  (or set KAGGLE_CONFIG_DIR)
     On Colab you can upload it and this script will place it for you.

Usage:
  python data/download_asl_alphabet.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "grassknoted/asl-alphabet"
DATA_DIR = Path(__file__).resolve().parent
ZIP_PATH = DATA_DIR / "asl-alphabet.zip"


def ensure_kaggle_credentials() -> None:
    home_token = Path.home() / ".kaggle" / "kaggle.json"
    local_token = DATA_DIR.parent / "kaggle.json"
    if home_token.exists():
        return
    if local_token.exists():
        home_token.parent.mkdir(parents=True, exist_ok=True)
        home_token.write_bytes(local_token.read_bytes())
        os.chmod(home_token, 0o600)
        print(f"Copied {local_token} -> {home_token}")
        return
    sys.exit(
        "No kaggle.json found. Create one at kaggle.com (Account -> API) and put it "
        "at ~/.kaggle/kaggle.json or in the repo root."
    )


def main() -> None:
    ensure_kaggle_credentials()
    print(f"Downloading {DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
        check=True,
    )
    print("Unzipping ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    # The archive contains asl_alphabet_train/asl_alphabet_train/<CLASS>/...
    # Flatten one level if needed so config paths line up.
    nested = DATA_DIR / "asl_alphabet_train" / "asl_alphabet_train"
    flat = DATA_DIR / "asl_alphabet_train"
    if nested.exists() and nested.is_dir():
        for child in nested.iterdir():
            child.rename(flat / child.name)
        nested.rmdir()

    print("Done. Train dir:", flat)
    print("Remember: the data/ folder is gitignored — never commit the images.")


if __name__ == "__main__":
    main()
