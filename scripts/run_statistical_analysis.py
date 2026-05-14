from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "linguistic_features_balanced.csv"

OUT_DIR = ROOT / "outputs" / "statistical_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = [
    "dataset",
    "domain",
    "label",
    "source",
    "text"
]


def cohens_d(x1, x2):
    n1 = len(x1)
    n2 = len(x2)

    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return 0

    return (np.mean(x1) - np.mean(x2)) / pooled_sd


def main():
    print("Loading feature dataset...")
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]

    human = df[df["label"] == 0]
    ai = df[df["label"] == 1]

    print(f"Human rows: {len(human):,}")
    print(f"AI rows: {len(ai):,}")
    print(f"Features: {len(feature_cols)}")

    rows = []

    for feature in feature_cols:
        human_values = human[feature].replace([np.inf, -np.inf], np.nan).dropna()
        ai_values = ai[feature].replace([np.inf, -np.inf], np.nan).dropna()

        t_stat, p_value = ttest_ind(
            human_values,
            ai_values,
            equal_var=False
        )

        d = cohens_d(human_values, ai_values)

        rows.append({
            "feature": feature,
            "human_mean": human_values.mean(),
            "human_sd": human_values.std(),
            "ai_mean": ai_values.mean(),
            "ai_sd": ai_values.std(),
            "mean_difference_human_minus_ai": human_values.mean() - ai_values.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": d,
            "higher_group": "human" if human_values.mean() > ai_values.mean() else "ai"
        })

    results = pd.DataFrame(rows)

    results["abs_cohens_d"] = results["cohens_d"].abs()
    results = results.sort_values("abs_cohens_d", ascending=False)

    out_csv = OUT_DIR / "human_vs_ai_linguistic_statistics.csv"
    results.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nSaved statistical table:")
    print(out_csv)

    print("\nTop features by absolute effect size:")
    print(results.head(20))

    # Bar chart of top 15 effect sizes
    top = results.head(15).copy()

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["abs_cohens_d"])
    plt.xlabel("Absolute Cohen's d")
    plt.ylabel("Feature")
    plt.title("Top linguistic differences between human and AI texts")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_effect_sizes.png", dpi=300)
    plt.close()

    print("\nSaved plot:")
    print(OUT_DIR / "top_effect_sizes.png")


if __name__ == "__main__":
    main()