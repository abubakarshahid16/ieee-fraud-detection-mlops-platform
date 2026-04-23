from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


def load_ieee_data(
    transaction_path: str | Path,
    identity_path: str | Path | None,
    label_column: str,
) -> DatasetBundle:
    transactions = pd.read_csv(transaction_path)
    if identity_path and Path(identity_path).exists():
        identity = pd.read_csv(identity_path)
        data = transactions.merge(identity, on="TransactionID", how="left")
    else:
        data = transactions

    labels = data[label_column].copy()
    features = data.drop(columns=[label_column])
    return DatasetBundle(features=features, labels=labels)


def time_based_split(features: pd.DataFrame, labels: pd.Series, train_fraction: float):
    split_index = int(len(features) * train_fraction)
    x_train = features.iloc[:split_index].reset_index(drop=True)
    y_train = labels.iloc[:split_index].reset_index(drop=True)
    x_test = features.iloc[split_index:].reset_index(drop=True)
    y_test = labels.iloc[split_index:].reset_index(drop=True)
    return x_train, x_test, y_train, y_test
