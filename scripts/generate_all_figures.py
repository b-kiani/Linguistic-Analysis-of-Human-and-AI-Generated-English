from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

HC3_PATH = ROOT / "data" / "processed" / "hc3_binary_texts.csv"
BAWE_PATH = ROOT / "data" / "processed" / "bawe_human_academic_texts.csv"
UNIFIED_PATH = ROOT / "data" / "processed" / "unified_human_ai_dataset_balanced.csv"
FEATURE_PATH = ROOT / "data" / "processed" / "linguistic_features_balanced.csv"

ML_PATH = ROOT / "outputs" / "ml_results" / "model_metrics_summary.csv"
STATS_PATH = ROOT / "outputs" / "statistical_analysis" / "human_vs_ai_linguistic_statistics.csv"
SHAP_PATH = ROOT / "outputs" / "shap_results" / "shap_feature_importance.csv"

OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_bar(labels, values, title, ylabel, out_path, rotation=0):
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_dataset_source_distribution():
    df = pd.read_csv(UNIFIED_PATH)

    counts = df["source"].value_counts()

    save_bar(
        counts.index,
        counts.values,
        "Balanced Dataset Composition by Source",
        "Number of texts",
        OUT_DIR / "Figure_1_dataset_source_distribution.png"
    )


def figure_dataset_domain_distribution():
    df = pd.read_csv(UNIFIED_PATH)

    counts = df["domain"].value_counts()

    save_bar(
        counts.index,
        counts.values,
        "Dataset Composition by Domain",
        "Number of texts",
        OUT_DIR / "Figure_2_dataset_domain_distribution.png",
        rotation=30
    )


def figure_hc3_domain_source_distribution():
    df = pd.read_csv(HC3_PATH)

    pivot = df.pivot_table(
        index="domain",
        columns="source",
        values="text",
        aggfunc="count",
        fill_value=0
    )

    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("HC3 Domain Distribution by Human and AI Source")
    plt.ylabel("Number of texts")
    plt.xlabel("Domain")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Figure_3_hc3_domain_source_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_model_performance_grouped():
    df = pd.read_csv(ML_PATH)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(df["model"]))
    width = 0.15

    plt.figure(figsize=(11, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, df[metric], width, label=metric)

    plt.xticks(x + width * 2, df["model"])
    plt.ylim(0.85, 1.00)
    plt.ylabel("Score")
    plt.title("Machine Learning Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Figure_4_model_performance_grouped.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_top_effect_sizes():
    df = pd.read_csv(STATS_PATH)

    top = df.head(15).copy()
    top["abs_cohens_d"] = top["cohens_d"].abs()

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["abs_cohens_d"])
    plt.xlabel("Absolute Cohen's d")
    plt.title("Top Linguistic Differences Between Human and AI Texts")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Figure_5_top_effect_sizes.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_top_shap_importance():
    df = pd.read_csv(SHAP_PATH)

    top = df.head(15).copy()

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top SHAP Feature Importance Values")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Figure_6_top_shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_key_feature_comparison():
    df = pd.read_csv(FEATURE_PATH)

    key_features = [
        "type_token_ratio",
        "avg_sentence_length",
        "flesch_kincaid_grade",
        "flesch_reading_ease",
        "passive_ratio",
        "discourse_marker_ratio"
    ]

    summary_rows = []

    for feature in key_features:
        human_mean = df[df["label"] == 0][feature].mean()
        ai_mean = df[df["label"] == 1][feature].mean()

        summary_rows.append({
            "feature": feature,
            "human": human_mean,
            "ai": ai_mean
        })

    summary = pd.DataFrame(summary_rows)

    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, summary["human"], width, label="Human")
    plt.bar(x + width / 2, summary["ai"], width, label="AI")

    plt.xticks(x, summary["feature"], rotation=30, ha="right")
    plt.ylabel("Mean value")
    plt.title("Comparison of Key Linguistic Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Figure_7_key_feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_pipeline_diagram():
    steps = [
        "Open-source datasets\nHC3 + BAWE",
        "Text preprocessing\ncleaning + duplicate removal",
        "Balanced dataset\n26,051 human + 26,051 AI",
        "Linguistic features\n22 interpretable variables",
        "Statistical analysis\nWelch t-tests + Cohen's d",
        "Machine learning\nRF + SVM + XGBoost",
        "Explainability\nSHAP analysis",
        "Linguistic interpretation\npublication-ready outputs"
    ]

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.axis("off")

    x_positions = np.linspace(0.05, 0.95, 4)
    y_positions = [0.70, 0.30]

    positions = []
    for y in y_positions:
        for x in x_positions:
            positions.append((x, y))

    for i, (step, (x, y)) in enumerate(zip(steps, positions)):
        ax.text(
            x,
            y,
            step,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white")
        )

        if i < len(steps) - 1:
            x2, y2 = positions[i + 1]

            if y == y2:
                ax.annotate(
                    "",
                    xy=(x2 - 0.08, y2),
                    xytext=(x + 0.08, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5)
                )
            else:
                ax.annotate(
                    "",
                    xy=(x2, y2 + 0.08),
                    xytext=(x, y - 0.08),
                    arrowprops=dict(arrowstyle="->", lw=1.5)
                )

    plt.title("Corpus-Based Explainable Machine Learning Pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Figure_8_pipeline_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def export_dataset_summary_table():
    hc3 = pd.read_csv(HC3_PATH)
    bawe = pd.read_csv(BAWE_PATH)
    unified = pd.read_csv(UNIFIED_PATH)
    features = pd.read_csv(FEATURE_PATH)

    rows = [
        {
            "Dataset": "HC3 processed",
            "Purpose": "Human and ChatGPT paired response dataset",
            "Rows": len(hc3),
            "Human texts": int((hc3["label"] == 0).sum()),
            "AI texts": int((hc3["label"] == 1).sum()),
            "Domains": ", ".join(sorted(hc3["domain"].unique()))
        },
        {
            "Dataset": "BAWE processed",
            "Purpose": "Additional authentic human academic writing",
            "Rows": len(bawe),
            "Human texts": len(bawe),
            "AI texts": 0,
            "Domains": "academic"
        },
        {
            "Dataset": "Balanced experimental dataset",
            "Purpose": "Final machine learning and linguistic analysis dataset",
            "Rows": len(unified),
            "Human texts": int((unified["label"] == 0).sum()),
            "AI texts": int((unified["label"] == 1).sum()),
            "Domains": ", ".join(sorted(unified["domain"].unique()))
        },
        {
            "Dataset": "Feature matrix",
            "Purpose": "Dataset after linguistic feature extraction",
            "Rows": len(features),
            "Human texts": int((features["label"] == 0).sum()),
            "AI texts": int((features["label"] == 1).sum()),
            "Domains": ", ".join(sorted(features["domain"].unique()))
        }
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "Dataset_summary_table.csv", index=False, encoding="utf-8-sig")

    print("\nDataset summary:")
    print(summary)


def main():
    figure_dataset_source_distribution()
    figure_dataset_domain_distribution()
    figure_hc3_domain_source_distribution()
    figure_model_performance_grouped()
    figure_top_effect_sizes()
    figure_top_shap_importance()
    figure_key_feature_comparison()
    figure_pipeline_diagram()
    export_dataset_summary_table()

    print("\nSaved all figures to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()