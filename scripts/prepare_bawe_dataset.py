from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]

BAWE_DIR = ROOT / "data" / "raw" / "bawe" / "extracted" / "download" / "CORPUS_ASCII"
OUT_PATH = ROOT / "data" / "processed" / "bawe_human_academic_texts.csv"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\ufeff", "")
    return text.strip()


def extract_text_from_xml(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "xml")

    # Remove metadata/header-like content if present
    for tag in soup.find_all(["teiHeader", "header", "metadata"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return clean_text(text)


def main():
    if not BAWE_DIR.exists():
        raise FileNotFoundError(f"BAWE folder not found: {BAWE_DIR}")

    rows = []
    xml_files = sorted(BAWE_DIR.glob("*.xml"))

    print(f"Found {len(xml_files):,} BAWE XML files")

    for path in xml_files:
        text = extract_text_from_xml(path)

        # Skip extremely short or failed extractions
        if len(text.split()) < 50:
            continue

        rows.append({
            "dataset": "BAWE",
            "domain": "academic",
            "label": 0,
            "source": "human",
            "file_id": path.stem,
            "text": text,
            "word_count": len(text.split())
        })

    df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(df):,}")

    print("\nWord count summary:")
    print(df["word_count"].describe())

    print("\nExample:")
    print(df[["file_id", "word_count", "text"]].head(3))


if __name__ == "__main__":
    main()