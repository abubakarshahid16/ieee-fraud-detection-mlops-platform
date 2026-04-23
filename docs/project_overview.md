# Complete Project Overview

## Project Name

IEEE-CIS Fraud Detection MLOps Platform

## Problem Solved

Financial fraud detection is not only a machine-learning problem. A real fraud system must detect suspicious transactions with high recall, handle severe class imbalance, scale under large transaction volume, and respond automatically when model performance or data quality degrades.

This project solves that full production problem by building a complete MLOps platform around the IEEE-CIS Fraud Detection dataset.

## End-to-End Solution

The system starts from raw IEEE-CIS transaction and identity data, validates and preprocesses the data, engineers features, trains multiple fraud-detection models, compares imbalance and cost-sensitive learning strategies, evaluates business-critical metrics, and then deploys the inference service with monitoring and automated retraining triggers.

## Full Workflow

1. Raw IEEE-CIS transaction and identity CSV files are loaded.
2. Data is merged using `TransactionID`.
3. Missing values are handled with scalable imputation strategies.
4. High-cardinality categorical features are frequency encoded.
5. Low-cardinality categorical features are one-hot encoded.
6. Class imbalance is handled using both `class_weight` and `SMOTE`.
7. XGBoost, LightGBM, and hybrid Random Forest models are trained.
8. Standard training is compared with cost-sensitive training.
9. Models are evaluated using precision, recall, F1-score, AUC-ROC, and confusion matrix.
10. Drift is simulated using time-based distribution shifts.
11. SHAP explainability is generated for fraud prediction interpretation.
12. The inference model is served through FastAPI.
13. Docker images are built for training and inference.
14. Kubernetes manifests deploy the service in an isolated namespace.
15. Kubeflow Pipelines orchestrates the ML workflow.
16. Prometheus collects system, model, and data metrics.
17. Grafana visualizes system health, model performance, and data drift.
18. Prometheus alerts detect recall drops, drift, and latency spikes.
19. Alertmanager forwards critical alerts to Jenkins.
20. Jenkins runs the CI/CD retraining workflow.

## Main Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- SMOTE
- SHAP
- FastAPI
- Docker
- Kubernetes
- Kubeflow Pipelines
- Prometheus
- Alertmanager
- Grafana
- Jenkins
- GitHub Actions

## What Makes This Project Valuable

This project demonstrates a production-style fraud detection system rather than a simple notebook. It includes the ML model, deployment layer, monitoring layer, alerting layer, CI/CD workflow, and retraining strategy.

The most important business value is reducing missed fraud through high recall while still documenting the false-positive trade-off.

## Evidence Included

The repository includes screenshots, logs, and reports proving:

- Real IEEE-CIS data was used.
- Models were trained and compared.
- Cost-sensitive learning was evaluated.
- Drift was simulated and measured.
- SHAP explainability was generated.
- Docker images were built.
- Kubernetes deployment succeeded.
- Kubeflow Pipelines was installed and a pipeline was submitted.
- Prometheus and Grafana ran live.
- Prometheus alert fired.
- Alertmanager triggered Jenkins.
- Jenkins pipeline completed successfully.

## Final Outcome

The project delivers a complete fraud-detection MLOps platform that can be presented to an instructor, evaluator, or client as an end-to-end solution for fraud monitoring, deployment, and retraining automation.
