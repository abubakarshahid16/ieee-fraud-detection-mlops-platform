from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from fraud_mlops.data import load_ieee_data, time_based_split
from fraud_mlops.drift import compute_numeric_drift, simulate_time_based_drift
from fraud_mlops.explainability import generate_shap_summary
from fraud_mlops.features import fit_transform_features, select_features
from fraud_mlops.modeling import (
    TrainingResult,
    apply_imbalance_strategy,
    build_model,
    evaluate_binary_classifier,
)
from fraud_mlops.monitoring_metrics import (
    DATA_DRIFT_SCORE,
    MODEL_FALSE_POSITIVE_RATE,
    MODEL_PRECISION,
    MODEL_RECALL,
)
from fraud_mlops.utils import ensure_dir, read_yaml, write_json


def _safe_feature_names(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns]


def run_training(config_path: str):
    config = read_yaml(config_path)
    artifacts_dir = ensure_dir(config["artifacts_dir"])
    bundle = load_ieee_data(
        transaction_path="data/train_transaction.csv",
        identity_path="data/train_identity.csv",
        label_column=config["label_column"],
    )
    features = bundle.features.drop(columns=config.get("drop_columns", []), errors="ignore")
    x_train_df, x_test_df, y_train, y_test = time_based_split(
        features, bundle.labels, config["train_fraction_for_time_split"]
    )
    x_train, x_test, feature_artifacts, retained_feature_names = fit_transform_features(
        x_train_df,
        x_test_df,
        config["top_high_cardinality"],
        config["max_missing_ratio"],
        config["max_numeric_features"],
        config["max_categorical_features"],
    )

    results: list[TrainingResult] = []
    best_result: TrainingResult | None = None
    best_score = -1.0
    selector = None
    best_selector = None

    for imbalance_strategy in config["imbalance_strategies"]:
        x_balanced, y_balanced = apply_imbalance_strategy(
            x_train,
            y_train,
            strategy=imbalance_strategy,
            random_state=config["random_state"],
            smote_ratio=config["smote_sampling_strategy"],
        )
        x_balanced_selected, x_test_selected, selector = select_features(
            x_balanced, y_balanced, x_test
        )

        for cost_strategy, penalty in config["cost_sensitive_multipliers"].items():
            for model_name in config["models"]:
                model = build_model(model_name, false_negative_penalty=penalty)
                fit_kwargs = {}
                if imbalance_strategy == "class_weight" and model_name == "xgboost":
                    fit_kwargs["sample_weight"] = [penalty if value == 1 else 1.0 for value in y_balanced]
                model.fit(x_balanced_selected, y_balanced, **fit_kwargs)
                metrics = evaluate_binary_classifier(
                    model, x_test_selected, y_test, config["evaluation_threshold"]
                )
                result = TrainingResult(
                    model_name=model_name,
                    imbalance_strategy=imbalance_strategy,
                    cost_strategy=cost_strategy,
                    model=model,
                    metrics=metrics,
                )
                results.append(result)
                score = metrics["auc_roc"] + metrics["recall"]
                if score > best_score:
                    best_result = result
                    best_score = score
                    best_selector = selector

    if best_result is None or best_selector is None:
        raise RuntimeError("Training failed to produce a model.")

    MODEL_RECALL.set(best_result.metrics["recall"])
    MODEL_PRECISION.set(best_result.metrics["precision"])
    MODEL_FALSE_POSITIVE_RATE.set(best_result.metrics["false_positive_rate"])

    drifted_test = simulate_time_based_drift(x_test_df)
    drift_result = compute_numeric_drift(x_test_df, drifted_test)
    DATA_DRIFT_SCORE.set(drift_result.overall_drift_score)

    summary = {
        "best_model": best_result.model_name,
        "best_imbalance_strategy": best_result.imbalance_strategy,
        "best_cost_strategy": best_result.cost_strategy,
        "best_metrics": best_result.metrics,
        "all_experiments": [
            {
                "model": result.model_name,
                "imbalance_strategy": result.imbalance_strategy,
                "cost_strategy": result.cost_strategy,
                "metrics": result.metrics,
            }
            for result in results
        ],
        "drift_summary": {
            "overall_drift_score": drift_result.overall_drift_score,
            "drifted_features": drift_result.drifted_features[:15],
        },
        "deployment_decision": best_result.metrics["auc_roc"] >= config["deploy_auc_threshold"],
    }
    write_json(summary, artifacts_dir / "metrics_summary.json")

    bundle_to_save = {
        "model": best_result.model,
        "threshold": config["evaluation_threshold"],
        "preprocessor": feature_artifacts.pipeline,
        "selector": best_selector,
        "feature_names": retained_feature_names,
        "metadata": summary,
    }
    joblib.dump(bundle_to_save, artifacts_dir / "best_model.joblib")

    sample_size = min(len(x_test_selected), 200)
    if sample_size > 0 and hasattr(best_result.model, "predict_proba"):
        try:
            generate_shap_summary(
                best_result.model,
                x_test_selected[:sample_size],
                artifacts_dir / "shap_summary.png",
            )
        except Exception as exc:  # pragma: no cover
            write_json({"shap_error": str(exc)}, artifacts_dir / "shap_error.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    summary = run_training(args.config)
    print(summary)


if __name__ == "__main__":
    main()
