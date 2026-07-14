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
import shutil
import sys
from pathlib import Path

DATASET = "grassknoted/asl-alphabet"
DATA_DIR = Path(__file__).resolve().parent


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
    print(f"Downloading {DATASET} (~1GB) ...")
    # Kaggle Python API (works regardless of PATH / venv activation, unlike the CLI).
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True, quiet=False)

    # The archive nests as asl_alphabet_train/asl_alphabet_train/<CLASS>/... Isolate the
    # real class dirs and drop anything else already at the flat path (e.g. stub data),
    # so the flatten is robust and idempotent.
    flat = DATA_DIR / "asl_alphabet_train"
    nested = flat / "asl_alphabet_train"
    if nested.is_dir():
        real = DATA_DIR / "asl_alphabet_train_real"
        if real.exists():
            shutil.rmtree(real)
        nested.rename(real)          # move real class dirs out
        shutil.rmtree(flat)          # remove the (now stub-only) flat dir
        real.rename(flat)            # put real data in place

    print("Done. Train dir:", flat)
    print("Remember: the data/ folder is gitignored — never commit the images.")


if __name__ == "__main__":
    main()
