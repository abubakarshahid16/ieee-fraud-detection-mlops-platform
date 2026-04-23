from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
EVIDENCE_DIR = ROOT / "docs" / "evidence"
ARTIFACTS = ROOT / "artifacts"
LOGS = EVIDENCE_DIR / "logs"


def read_text_any(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def save_text_figure(title: str, lines: list[str], output_name: str, font_size: int = 11) -> None:
    height = max(4, 0.32 * len(lines) + 1.5)
    fig, ax = plt.subplots(figsize=(13, height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left", pad=16)
    y = 0.98
    for line in lines:
        ax.text(0.01, y, line, va="top", ha="left", family="monospace", fontsize=font_size)
        y -= 0.06
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(metrics: dict) -> None:
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Fraud Class Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "03_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_model_comparison(experiments: list[dict]) -> None:
    labels = [
        f"{row['model']}\n{row['imbalance_strategy']}\n{row['cost_strategy']}"
        for row in experiments
    ]
    recall = [row["metrics"]["recall"] for row in experiments]
    auc = [row["metrics"]["auc_roc"] for row in experiments]

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, recall, width, label="Recall")
    ax.bar(x + width / 2, auc, width, label="AUC-ROC")
    ax.set_title("Model and Strategy Comparison")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "04_model_comparison.png", dpi=180)
    plt.close(fig)


def save_drift_summary(summary: dict) -> None:
    drifted = summary["drift_summary"]["drifted_features"][:10]
    labels = [row["feature"] for row in drifted]
    values = [row["ks_statistic"] for row in drifted]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color="#e76f51")
    ax.set_title("Top Drifted Features")
    ax.set_xlabel("KS Statistic")
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "05_drift_summary.png", dpi=180)
    plt.close(fig)


def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((ARTIFACTS / "metrics_summary.json").read_text(encoding="utf-8"))
    save_text_figure(
        "Real IEEE-CIS Dataset Summary",
        read_text_any(LOGS / "data_generation.log").splitlines(),
        "01_data_generation.png",
    )
    save_text_figure(
        "Training Summary",
        json.dumps(metrics, indent=2).splitlines()[:40],
        "02_training_summary.png",
        font_size=10,
    )
    save_confusion_matrix(metrics["best_metrics"])
    save_model_comparison(metrics["all_experiments"])
    save_drift_summary(metrics)

    for log_name, output_name, title in [
        ("kubeflow_compile.log", "06_kubeflow_compile.png", "Kubeflow Pipeline Compilation"),
        ("k8s_validation.log", "07_k8s_validation.png", "Kubernetes Manifest Validation"),
        ("tests_and_ci.log", "08_tests_and_ci.png", "Tests and Local CI Checks"),
        ("api_prediction.log", "09_api_prediction.png", "Inference API Prediction"),
        ("monitoring_overview.log", "10_monitoring_overview.png", "Monitoring and Dashboards"),
        ("alert_trigger.log", "11_alert_trigger.png", "Alert Trigger and Intelligent Retraining"),
        ("submission_tree.log", "12_submission_tree.png", "Final Submission Tree"),
    ]:
        save_text_figure(
            title,
            read_text_any(LOGS / log_name).splitlines(),
            output_name,
            font_size=10,
        )

    dataset = pd.read_csv(ROOT / "data" / "train_transaction.csv", nrows=200)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dataset["TransactionAmt"], bins=30, color="#2a9d8f", edgecolor="white")
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("TransactionAmt")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "13_transaction_distribution.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
