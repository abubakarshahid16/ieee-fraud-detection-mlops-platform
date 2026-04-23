# Requirement Coverage

## Status Summary

This file maps the assignment requirements to the implemented artifacts and live evidence collected in this repository.

## Task 1: Kubeflow Environment Setup

### Requirement

- Deploy Kubeflow
- Configure persistent volumes
- Configure resource quotas
- Create isolated namespace
- Build Kubeflow pipeline with:
  - Data Ingestion
  - Data Validation
  - Data Preprocessing
  - Feature Engineering
  - Model Training
  - Model Evaluation
  - Conditional Deployment
- Add retry logic

### Coverage

- Kubeflow pipeline definition: `kubeflow/pipeline.py`
- Compiled pipeline artifact: `artifacts/fraud_detection_pipeline.yaml`
- Namespace manifest: `k8s/namespace.yaml`
- Resource quota manifest: `k8s/resource-quota.yaml`
- Persistent volume manifest: `k8s/persistent-volume.yaml`
- Persistent volume claim manifest: `k8s/persistent-volume-claim.yaml`
- Live Kubernetes namespace and rollout evidence:
  - `docs/evidence/logs/k8s_namespace_live.yaml`
  - `docs/evidence/logs/k8s_quota_live.yaml`
  - `docs/evidence/logs/k8s_rollout_live.txt`

### Live Evidence

- Kubeflow Pipelines installed in the `kubeflow` namespace on the live kind cluster.
- Kubeflow Pipelines UI returned HTTP `200` through port-forward.
- The compiled fraud pipeline was submitted to the live Kubeflow Pipelines API.
- Evidence files:
  - `docs/evidence/logs/kubeflow_pods_live.txt`
  - `docs/evidence/logs/kubeflow_services_live.txt`
  - `docs/evidence/logs/kubeflow_ui_status.txt`
  - `docs/evidence/logs/kubeflow_pipeline_run_submit.log`
  - `docs/evidence/logs/kubeflow_pipeline_run_status.log`

## Task 2: Data Challenges Handling

### Coverage

- Missing values handled in `src/fraud_mlops/features.py`
- High-cardinality handling with frequency encoding in `src/fraud_mlops/features.py`
- Two imbalance strategies compared:
  - `class_weight`
  - `SMOTE`
- Real-data experiment results: `artifacts/metrics_summary.json`

## Task 3: Model Complexity

### Coverage

- XGBoost: `src/fraud_mlops/modeling.py`
- LightGBM: `src/fraud_mlops/modeling.py`
- Hybrid RF + feature selection:
  - `src/fraud_mlops/modeling.py`
  - `src/fraud_mlops/features.py`

### Metrics

- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion matrix

Stored in `artifacts/metrics_summary.json`.

## Task 4: Cost-Sensitive Learning

### Coverage

- Standard vs cost-sensitive training comparison in `src/fraud_mlops/train.py`
- Real results captured in `artifacts/metrics_summary.json`
- Business-impact discussion in:
  - `docs/research_report.md`
  - `docs/research_report.docx`

## Task 5: CI/CD Pipeline with Intelligent Triggers

### Coverage

- Workflow file: `.github/workflows/fraud-mlops-cicd.yml`
- Docker build evidence:
  - `docs/evidence/logs/docker_build_training.log`
  - `docs/evidence/logs/docker_build_api.log`
- Local registry push evidence:
  - `docs/evidence/logs/docker_push_training.log`
  - `docs/evidence/logs/docker_push_api.log`
- Intelligent trigger logic:
  - `src/fraud_mlops/intelligent_trigger.py`
  - `artifacts/live_alert_retraining_event.json`
- Live alert-to-trigger evidence:
  - `docs/evidence/logs/prometheus_alerts.json`
  - `docs/evidence/logs/live_alert_to_retraining.log`

### Live Evidence

- A real Jenkins server was run locally in Docker.
- A Jenkins pipeline job named `fraud-mlops-pipeline` was created.
- The Jenkins job was triggered through a webhook endpoint.
- Alertmanager was configured to call the Jenkins webhook.
- Console logs were captured from successful pipeline runs.
- Evidence files:
  - `docs/evidence/logs/jenkins_job.json`
  - `docs/evidence/logs/jenkins_console.log`
  - `docs/evidence/logs/jenkins_alertmanager_console.log`
  - `docs/evidence/logs/jenkins_alertmanager_last_build.json`
  - `monitoring/alertmanager-local.yml`

## Task 6: Observability and Monitoring

### Coverage

- Prometheus config: `monitoring/prometheus.yml`
- Local Prometheus config used for live run: `monitoring/prometheus-local.yml`
- Alert rules: `monitoring/alert_rules.yml`
- Grafana dashboards:
  - `monitoring/grafana/dashboards/system-health.json`
  - `monitoring/grafana/dashboards/model-performance.json`
  - `monitoring/grafana/dashboards/data-drift.json`
- Live monitoring evidence:
  - `docs/evidence/logs/prometheus_health.txt`
  - `docs/evidence/logs/prometheus_targets.json`
  - `docs/evidence/logs/prometheus_query_requests.json`
  - `docs/evidence/logs/grafana_health.json`
  - `docs/evidence/logs/grafana_dashboards.json`
  - `docs/evidence/logs/alertmanager_alerts.json`
  - `docs/evidence/logs/jenkins_alertmanager_console.log`

## Task 7: Drift Simulation

### Coverage

- Drift simulation and measurement: `src/fraud_mlops/drift.py`
- Real drift results: `artifacts/metrics_summary.json`

## Task 8: Intelligent Retraining Strategy

### Coverage

- Threshold-based retraining strategy implemented
- Trigger helper updated for both `gte` and `lte` comparisons
- Live alert-driven retraining event artifact:
  - `artifacts/live_alert_retraining_event.json`

## Task 9: Explainability

### Coverage

- SHAP generation: `src/fraud_mlops/explainability.py`
- Artifact: `artifacts/shap_summary.png`

## Report and Screenshots

### Coverage

- Word report: `docs/research_report.docx`
- Markdown report: `docs/research_report.md`
- Evidence images:
  - `docs/evidence/*.png`

Each evidence image is included in the Word report with explanatory context.
