from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HC3_PATH = ROOT / "data" / "processed" / "hc3_binary_texts.csv"
BAWE_PATH = ROOT / "data" / "processed" / "bawe_human_academic_texts.csv"

OUT_ALL = ROOT / "data" / "processed" / "unified_human_ai_dataset.csv"
OUT_BALANCED = ROOT / "data" / "processed" / "unified_human_ai_dataset_balanced.csv"


def main():
    print("Loading HC3...")
    hc3 = pd.read_csv(HC3_PATH)

    print("Loading BAWE...")
    bawe = pd.read_csv(BAWE_PATH)

    # Standardise columns
    hc3 = hc3[["dataset", "domain", "label", "source", "text"]].copy()
    bawe = bawe[["dataset", "domain", "label", "source", "text"]].copy()

    # Remove missing or empty texts
    hc3 = hc3.dropna(subset=["text"])
    bawe = bawe.dropna(subset=["text"])

    hc3["text"] = hc3["text"].astype(str).str.strip()
    bawe["text"] = bawe["text"].astype(str).str.strip()

    hc3 = hc3[hc3["text"].str.split().str.len() >= 30]
    bawe = bawe[bawe["text"].str.split().str.len() >= 30]

    # Combine HC3 + BAWE
    df = pd.concat([hc3, bawe], ignore_index=True)

    # Remove exact duplicate texts
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(df)

    print(f"Removed duplicate rows: {before - after:,}")

    df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

    print(f"\nSaved full unified dataset:")
    print(OUT_ALL)
    print(f"Rows: {len(df):,}")

    print("\nFull dataset label counts:")
    print(df["label"].value_counts())

    print("\nFull dataset source counts:")
    print(df["source"].value_counts())

    print("\nFull dataset domain counts:")
    print(df["domain"].value_counts())

    # Create balanced dataset for fair ML classification
    human_df = df[df["label"] == 0]
    ai_df = df[df["label"] == 1]

    n = min(len(human_df), len(ai_df))

    balanced = pd.concat([
        human_df.sample(n=n, random_state=42),
        ai_df.sample(n=n, random_state=42)
    ], ignore_index=True)

    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    balanced.to_csv(OUT_BALANCED, index=False, encoding="utf-8-sig")

    print(f"\nSaved balanced unified dataset:")
    print(OUT_BALANCED)
    print(f"Rows: {len(balanced):,}")

    print("\nBalanced dataset label counts:")
    print(balanced["label"].value_counts())

    print("\nBalanced dataset source counts:")
    print(balanced["source"].value_counts())

    print("\nBalanced dataset domain counts:")
    print(balanced["domain"].value_counts())


if __name__ == "__main__":
    main()