from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

hc3_path = ROOT / "data" / "processed" / "hc3_binary_texts.csv"
bawe_dir = ROOT / "data" / "raw" / "bawe" / "extracted"

print("=" * 80)
print("HC3 PROCESSED DATASET")
print("=" * 80)

if hc3_path.exists():
    df = pd.read_csv(hc3_path)
    print(f"File: {hc3_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\nSource counts:")
    print(df["source"].value_counts())
    print("\nDomain counts:")
    print(df["domain"].value_counts())
    print("\nExample rows:")
    print(df.head(3))
else:
    print("HC3 processed CSV not found.")

print("\n" + "=" * 80)
print("BAWE RAW DATASET")
print("=" * 80)

if bawe_dir.exists():
    files = list(bawe_dir.rglob("*"))
    text_like = [
        f for f in files
        if f.is_file() and f.suffix.lower() in [".txt", ".xml", ".html", ".htm"]
    ]

    print(f"BAWE folder: {bawe_dir}")
    print(f"Total files/folders found: {len(files):,}")
    print(f"Text/XML/HTML-like files found: {len(text_like):,}")

    print("\nFirst 20 text-like files:")
    for f in text_like[:20]:
        print(f.relative_to(bawe_dir))
else:
    print("BAWE extracted folder not found.")