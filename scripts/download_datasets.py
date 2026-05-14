"""
Download datasets for:
Explainable ML for Linguistic Analysis of Human and AI-Generated English.

This revised version avoids the Hugging Face `datasets` legacy-script error:
RuntimeError: Dataset scripts are no longer supported, but found HC3.py

Run from project root on Windows PowerShell:
    python scripts/download_datasets.py
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Correct HC3 config/file names from the HC3 repository.
HC3_FILES = {
    "all": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl",
    "reddit_eli5": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/reddit_eli5.jsonl",
    "wiki_csai": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/wiki_csai.jsonl",
    "open_qa": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/open_qa.jsonl",
    "finance": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/finance.jsonl",
    "medicine": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/medicine.jsonl",
}

BAWE_ZIP_URL = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2539/2539.zip?isAllowed=y&sequence=3"


def ensure_dirs() -> None:
    for p in [RAW / "hc3", RAW / "bawe", RAW / "toefl11", PROCESSED]:
        p.mkdir(parents=True, exist_ok=True)


def stream_download(url: str, dest: Path, overwrite: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"Already exists, skipping: {dest}")
        return True

    print(f"Downloading: {url}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                print(f"Could not download {url}. HTTP status: {r.status_code}")
                return False
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"Download failed for {url}: {exc}")
        return False


def download_hc3(overwrite: bool = False) -> None:
    out_dir = RAW / "hc3"
    success = 0
    for name, url in HC3_FILES.items():
        out_file = out_dir / f"hc3_{name}.jsonl"
        ok = stream_download(url, out_file, overwrite=overwrite)
        if ok:
            success += 1
            print(f"[HC3] Saved: {out_file}")
    print(f"[HC3] Finished: {success}/{len(HC3_FILES)} files available.")


def download_bawe(overwrite: bool = False) -> None:
    out_dir = RAW / "bawe"
    zip_path = out_dir / "2539.zip"
    ok = stream_download(BAWE_ZIP_URL, zip_path, overwrite=overwrite)
    if not ok:
        print("[BAWE] Automatic download failed. Download manually from Oxford Text Archive and place 2539.zip in data/raw/bawe/.")
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir / "extracted")
        print(f"[BAWE] Extracted to: {out_dir / 'extracted'}")
    except zipfile.BadZipFile:
        print("[BAWE] File is not a valid ZIP. It may be a login/terms HTML page. Download manually if needed.")


def write_toefl11_instructions() -> None:
    text = """TOEFL11 / ETS Corpus of Non-Native Written English

This dataset is distributed by the Linguistic Data Consortium as LDC2014T06.
It cannot be downloaded automatically unless you have LDC access and accept the licence.

After obtaining the corpus:
1. Place the unzipped corpus folder here: data/raw/toefl11/
2. Keep original and tokenized essay files in separate subfolders if provided.
3. Keep metadata/prompt files for stratified analysis.
"""
    path = RAW / "toefl11" / "README_TOEFL11_LICENSE_REQUIRED.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[TOEFL11] Wrote licence/download instructions: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hc3", action="store_true")
    parser.add_argument("--skip-bawe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if not args.skip_hc3:
        download_hc3(overwrite=args.overwrite)
    if not args.skip_bawe:
        download_bawe(overwrite=args.overwrite)
    write_toefl11_instructions()
    print("Dataset download step finished.")


if __name__ == "__main__":
    main()
