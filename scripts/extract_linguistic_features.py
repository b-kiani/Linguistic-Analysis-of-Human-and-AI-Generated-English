from pathlib import Path
import re
import pandas as pd
import spacy
import textstat
from tqdm import tqdm

tqdm.pandas()

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data" / "processed" / "unified_human_ai_dataset_balanced.csv"
OUT_PATH = ROOT / "data" / "processed" / "linguistic_features_balanced.csv"

nlp = spacy.load("en_core_web_sm", disable=["ner"])


DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "in conclusion", "for example", "for instance",
    "on the other hand", "in addition", "as a result", "firstly",
    "secondly", "finally", "overall"
]


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_divide(a, b):
    return a / b if b else 0


def extract_features(text):
    text = clean_text(text)
    doc = nlp(text)

    tokens = [
        token for token in doc
        if not token.is_space and not token.is_punct
    ]

    words = [
        token.text.lower() for token in tokens
        if token.is_alpha
    ]

    sentences = list(doc.sents)

    word_count = len(words)
    unique_words = len(set(words))
    sentence_count = len(sentences)

    avg_word_length = safe_divide(
        sum(len(w) for w in words),
        word_count
    )

    avg_sentence_length = safe_divide(word_count, sentence_count)

    type_token_ratio = safe_divide(unique_words, word_count)

    lexical_tokens = [
        token for token in tokens
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
    ]

    lexical_density = safe_divide(len(lexical_tokens), len(tokens))

    pos_counts = {}
    for pos in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ"]:
        pos_counts[f"pos_{pos.lower()}_ratio"] = safe_divide(
            sum(1 for token in tokens if token.pos_ == pos),
            len(tokens)
        )

    passive_count = 0
    for token in doc:
        if token.dep_ == "auxpass":
            passive_count += 1
        if token.dep_ == "nsubjpass":
            passive_count += 1

    passive_ratio = safe_divide(passive_count, sentence_count)

    lower_text = text.lower()
    discourse_marker_count = sum(
        lower_text.count(marker) for marker in DISCOURSE_MARKERS
    )

    discourse_marker_ratio = safe_divide(discourse_marker_count, word_count)

    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
    except Exception:
        flesch_reading_ease = 0

    try:
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    except Exception:
        flesch_kincaid_grade = 0

    return {
        "word_count": word_count,
        "unique_word_count": unique_words,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": type_token_ratio,
        "lexical_density": lexical_density,
        "passive_ratio": passive_ratio,
        "discourse_marker_count": discourse_marker_count,
        "discourse_marker_ratio": discourse_marker_ratio,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        **pos_counts
    }


def main():
    print("Loading dataset...")
    df = pd.read_csv(IN_PATH)

    print(f"Rows loaded: {len(df):,}")

    # Optional speed control for first test
    # Uncomment this line for a quick test first:
    # df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    print("Extracting linguistic features...")
    feature_rows = []

    for text in tqdm(df["text"], total=len(df)):
        feature_rows.append(extract_features(text))

    features = pd.DataFrame(feature_rows)

    out = pd.concat([
        df[["dataset", "domain", "label", "source", "text"]].reset_index(drop=True),
        features
    ], axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved:")
    print(OUT_PATH)
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns):,}")

    print("\nFeature columns:")
    print(list(features.columns))


if __name__ == "__main__":
    main()