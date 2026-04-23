from fraud_mlops.drift import compute_numeric_drift

import pandas as pd


def test_compute_numeric_drift_returns_non_negative_score():
    reference = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    current = pd.DataFrame({"a": [2, 3, 4], "b": [3, 4, 8]})
    result = compute_numeric_drift(reference, current)
    assert result.overall_drift_score >= 0
    assert len(result.drifted_features) == 2
