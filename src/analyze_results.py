#!/usr/bin/env python3
"""Aggregate five-fold results, run McNemar tests, and create final figures."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar


METRICS = [
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "auc",
    "loss",
]

DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv2_100": "MobileNetV2",
    "squeezenet1_0": "SqueezeNet 1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def load_oof(results_dir: Path, model: str, folds: int) -> pd.DataFrame:
    parts = [
        pd.read_csv(results_dir / "fold_predictions" / f"{model}_fold_{fold}_predictions.csv")
        for fold in range(1, folds + 1)
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("image_id")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    tables = args.results_dir / "tables"
    figures = args.results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.read_csv(args.results_dir / "metrics" / "cv_fold_metrics.csv")
    models = fold_metrics["model"].drop_duplicates().tolist()

    summary_rows = []
    for model, group in fold_metrics.groupby("model"):
        row = {"model": model}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "cv_summary_mean_sd.csv", index=False)

    def fmt(mean, sd):
        return f"{mean:.4f} ± {sd:.4f}"

    paper = pd.DataFrame({
        "Model": summary["model"].map(lambda x: DISPLAY_NAMES.get(x, x)),
        "Accuracy": [fmt(m, s) for m, s in zip(summary.accuracy_mean, summary.accuracy_sd)],
        "Precision": [fmt(m, s) for m, s in zip(summary.precision_mean, summary.precision_sd)],
        "Sensitivity": [fmt(m, s) for m, s in zip(summary.recall_sensitivity_mean, summary.recall_sensitivity_sd)],
        "Specificity": [fmt(m, s) for m, s in zip(summary.specificity_mean, summary.specificity_sd)],
        "F1": [fmt(m, s) for m, s in zip(summary.f1_mean, summary.f1_sd)],
        "AUROC": [fmt(m, s) for m, s in zip(summary.auc_mean, summary.auc_sd)],
    })
    paper.to_csv(tables / "paper_ready_metrics_table.csv", index=False)

    oof = {model: load_oof(args.results_dir, model, args.folds) for model in models}
    base_ids = oof[models[0]]["image_id"].tolist()
    for model in models[1:]:
        if oof[model]["image_id"].tolist() != base_ids:
            raise RuntimeError(f"Out-of-fold sample order differs for {model}.")

    mcnemar_rows = []
    for m1, m2 in combinations(models, 2):
        c1 = (oof[m1]["y_true"].to_numpy() == oof[m1]["y_pred"].to_numpy()).astype(int)
        c2 = (oof[m2]["y_true"].to_numpy() == oof[m2]["y_pred"].to_numpy()).astype(int)

        both_correct = int(((c1 == 1) & (c2 == 1)).sum())
        m1_only = int(((c1 == 1) & (c2 == 0)).sum())
        m2_only = int(((c1 == 0) & (c2 == 1)).sum())
        both_wrong = int(((c1 == 0) & (c2 == 0)).sum())

        result = mcnemar(
            [[both_correct, m1_only], [m2_only, both_wrong]],
            exact=False,
            correction=True,
        )
        mcnemar_rows.append({
            "model_1": m1,
            "model_2": m2,
            "both_correct": both_correct,
            "model_1_correct_model_2_wrong": m1_only,
            "model_1_wrong_model_2_correct": m2_only,
            "both_wrong": both_wrong,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        })

    pd.DataFrame(mcnemar_rows).to_csv(
        tables / "mcnemar_pairwise_results.csv", index=False
    )

    plt.figure(figsize=(8, 6))
    for model in models:
        df = oof[model]
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = roc_auc_score(df["y_true"], df["y_prob"])
        plt.plot(fpr, tpr, label=f"{DISPLAY_NAMES.get(model, model)} (AUROC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Out-of-Fold ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(figures / "final_cv_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    for model in models:
        df = oof[model]
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        image = ax.imshow(cm)
        ax.set_title(DISPLAY_NAMES.get(model, model))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], labels=["Non-referable", "Referable"])
        ax.set_yticks([0, 1], labels=["Non-referable", "Referable"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(
            figures / f"{model}_oof_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Wrote summary tables and figures under {args.results_dir}")


if __name__ == "__main__":
    main()
