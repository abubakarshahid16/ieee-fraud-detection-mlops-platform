from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def generate_shap_summary(model, x_sample, output_path: str | Path) -> None:
    explainer = shap.Explainer(model, x_sample)
    shap_values = explainer(x_sample)
    values = getattr(shap_values, "values", shap_values)
    if hasattr(values, "ndim") and values.ndim == 3:
        values = values[:, :, -1]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, x_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def top_feature_importances(model, feature_names: list[str], top_k: int = 15):
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    ranking = np.argsort(importances)[::-1][:top_k]
    return [
        {"feature": feature_names[index], "importance": float(importances[index])}
        for index in ranking
    ]
