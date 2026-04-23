from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API latency in seconds",
    ["endpoint"],
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Prediction confidence distribution",
    buckets=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
)
MODEL_RECALL = Gauge("fraud_model_recall", "Fraud recall tracked by monitoring")
MODEL_PRECISION = Gauge("fraud_model_precision", "Fraud precision tracked by monitoring")
MODEL_FALSE_POSITIVE_RATE = Gauge(
    "fraud_model_false_positive_rate", "Fraud model false positive rate"
)
DATA_DRIFT_SCORE = Gauge("fraud_data_drift_score", "Aggregated data drift score")
MISSING_VALUE_RATIO = Gauge("fraud_missing_value_ratio", "Input missing value ratio")
