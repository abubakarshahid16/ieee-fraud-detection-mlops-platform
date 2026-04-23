from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


@dataclass
class TrainingResult:
    model_name: str
    imbalance_strategy: str
    cost_strategy: str
    model: object
    metrics: dict


def apply_imbalance_strategy(x_train, y_train, strategy: str, random_state: int, smote_ratio: float):
    if strategy == "smote":
        sampler = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
        return sampler.fit_resample(x_train, y_train)
    if strategy == "class_weight":
        return x_train, y_train
    raise ValueError(f"Unsupported imbalance strategy: {strategy}")


def build_model(model_name: str, false_negative_penalty: float):
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            scale_pos_weight=false_negative_penalty,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=48,
            class_weight={0: 1.0, 1: false_negative_penalty},
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    if model_name == "hybrid_rf":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=5,
            class_weight={0: 1.0, 1: false_negative_penalty},
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_binary_classifier(model, x_test, y_test, threshold: float):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    confusion = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion,
        "false_positive_rate": float(confusion[0][1] / max(sum(confusion[0]), 1)),
        "fraud_detection_rate": float(np.mean(predictions)),
    }
