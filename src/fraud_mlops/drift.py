from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftResult:
    overall_drift_score: float
    drifted_features: list[dict]


def compute_numeric_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    numeric_columns = reference_df.select_dtypes(include=["number"]).columns
    details: list[dict] = []
    scores: list[float] = []
    for column in numeric_columns:
        ref_values = reference_df[column].dropna()
        cur_values = current_df[column].dropna()
        if ref_values.empty or cur_values.empty:
            continue
        statistic, p_value = ks_2samp(ref_values, cur_values)
        scores.append(float(statistic))
        details.append(
            {
                "feature": column,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < 0.05 and statistic > 0.1),
            }
        )
    overall = float(np.mean(scores)) if scores else 0.0
    return DriftResult(overall_drift_score=overall, drifted_features=details)


def simulate_time_based_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.copy()
    numeric_columns = drifted.select_dtypes(include=["number"]).columns.tolist()[:5]
    for index, column in enumerate(numeric_columns, start=1):
        drifted[column] = drifted[column].fillna(0) * (1 + index * 0.05)
    if "card4" in drifted.columns:
        drifted.loc[drifted.index[-len(drifted) // 5 :], "card4"] = "emerging_card_network"
    return drifted
