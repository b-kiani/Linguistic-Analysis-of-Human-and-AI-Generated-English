"""
Convert HC3 JSONL files into a flat binary classification CSV:
label=0 human, label=1 ai/chatgpt

Run after download_datasets.py:
    python scripts/prepare_hc3_binary_dataset.py
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HC3_DIR = ROOT / "data" / "raw" / "hc3"
OUT = ROOT / "data" / "processed" / "hc3_binary_texts.csv"


def iter_hc3_files():
    # Prefer the combined all file to avoid duplicated rows from the domain files.
    all_file = HC3_DIR / "hc3_all.jsonl"
    if all_file.exists() and all_file.stat().st_size > 0:
        yield all_file
    else:
        for fp in sorted(HC3_DIR.glob("hc3_*.jsonl")):
            if fp.stat().st_size > 0:
                yield fp


def get_domain(fp: Path, obj: dict) -> str:
    if fp.stem == "hc3_all":
        return str(obj.get("source") or "unknown")
    return fp.stem.replace("hc3_", "")


def main() -> None:
    rows = []
    files = list(iter_hc3_files())

    if not files:
        raise FileNotFoundError(
            f"No HC3 JSONL files found in {HC3_DIR}. Run: python scripts/download_datasets.py"
        )

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping invalid JSON in {fp.name}, line {line_no}: {exc}")
                    continue

                domain = get_domain(fp, obj)
                question = str(obj.get("question") or "").strip()

                for txt in obj.get("human_answers", []) or []:
                    if isinstance(txt, str) and txt.strip():
                        rows.append({
                            "dataset": "HC3", "domain": domain, "label": 0,
                            "source": "human", "question": question, "text": txt.strip()
                        })

                for txt in obj.get("chatgpt_answers", []) or []:
                    if isinstance(txt, str) and txt.strip():
                        rows.append({
                            "dataset": "HC3", "domain": domain, "label": 1,
                            "source": "ai_chatgpt", "question": question, "text": txt.strip()
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("HC3 files were found, but no usable texts were extracted. Check the JSONL structure.")

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"Saved {OUT} with {len(df):,} rows")
    print("\nCounts by domain and source:")
    print(df.groupby(["domain", "source"]).size().sort_index())
    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
