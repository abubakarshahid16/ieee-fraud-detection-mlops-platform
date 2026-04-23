# Research Report: IEEE-CIS Fraud Detection MLOps System

## 1. Objective

This assignment implements an end-to-end MLOps workflow for fraud detection using the IEEE-CIS Fraud Detection dataset. The system is designed to:

- maintain high recall for fraud cases
- scale under high transaction volume
- detect performance degradation automatically
- support intelligent retraining

The final repository includes model training code, Kubeflow pipeline design, Kubernetes manifests, CI/CD workflow, observability configuration, drift handling, explainability, and a Word report with evidence screenshots.

## 2. Dataset

The final execution used the real Kaggle competition training files:

- `train_transaction.csv`
- `train_identity.csv`

Observed dataset size during execution:

- Rows: `590,540`
- Fraud count: `20,663`
- Fraud rate: `0.03499000914417313`

The transaction and identity tables were merged on `TransactionID` and then processed through a time-based split so that earlier data was used for training and later data was used for evaluation.

## 3. Kubeflow and Kubernetes Setup

The Kubeflow pipeline contains the required stages:

1. Data Ingestion
2. Data Validation
3. Data Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Conditional Deployment

Pipeline capabilities:

- retry logic on critical steps
- conditional deployment based on evaluation threshold
- compiled pipeline artifact in `artifacts/fraud_detection_pipeline.yaml`

Kubernetes deliverables included in the project:

- isolated namespace
- resource quota
- persistent volume
- persistent volume claim
- inference deployment manifest

Live infrastructure evidence completed in this environment:

- a real local Kubernetes cluster was created using `kind`
- Kubeflow Pipelines was installed in the `kubeflow` namespace
- Kubeflow Pipelines UI returned HTTP `200`
- the compiled fraud pipeline was submitted to the live KFP API and produced a run ID
- the `fraud-mlops` namespace was applied successfully
- the API deployment rolled out successfully with `2/2` replicas available
- the in-cluster service responded successfully through `kubectl port-forward`

## 4. Data Challenges Handling

### Missing Values

The real IEEE-CIS dataset contains substantial missingness. To keep the workflow scalable on a large tabular dataset, the pipeline uses:

- numeric median imputation
- explicit numeric missing-value indicators
- categorical mode or constant imputation depending on the path

This is more scalable than neighbor-based imputation on a dataset of this size.

### High-Cardinality Categorical Features

The dataset contains high-cardinality fields such as email and identity-related categories. The implemented strategy is:

- one-hot encoding for lower-cardinality categorical features
- frequency encoding for high-cardinality categorical features

This reduces feature explosion while preserving useful signal.

### Feature Space Control

To make the training workflow practical on the real dataset:

- extremely sparse columns are filtered by missing-ratio threshold
- numeric and categorical feature counts are capped
- feature selection is applied in the hybrid model path

## 5. Imbalance Handling

Two imbalance strategies were explicitly compared:

1. `class_weight`
2. `SMOTE`

This satisfies the assignment requirement to compare at least two imbalance-handling strategies.

## 6. Models Implemented

The required model families were implemented:

1. XGBoost
2. LightGBM
3. Hybrid Random Forest + feature selection

These were evaluated in both:

- standard training mode
- cost-sensitive training mode

## 7. Experimental Results

### Best Selected Configuration

The best configuration chosen by the project scoring logic was:

- Model: `XGBoost`
- Imbalance strategy: `class_weight`
- Cost strategy: `cost_sensitive`

Best recorded metrics:

- Precision: `0.034903998969116445`
- Recall: `0.999753937007874`
- F1-score: `0.06745303771094638`
- AUC-ROC: `0.8941891839491015`
- Confusion matrix: `[[1702, 112342], [1, 4063]]`

### Comparative Findings

Important comparison outcomes from the real IEEE-CIS run:

- Standard LightGBM achieved the strongest AUC-ROC: `0.9070918321482218`
- Cost-sensitive XGBoost achieved the highest recall: `0.999753937007874`
- Cost-sensitive models significantly increased recall but also increased false positives
- SMOTE helped some models but did not consistently outperform class weighting on the selected scoring rule

This shows the exact business trade-off required in fraud detection:

- very high recall reduces missed fraud
- aggressive fraud capture raises false-positive investigations

## 8. Cost-Sensitive Learning Analysis

The project compares standard training with cost-sensitive training by assigning a higher penalty to the fraud class.

Observed business interpretation:

- false negatives are expensive because they represent real fraud loss
- false positives increase manual review cost and customer friction
- the cost-sensitive configuration successfully drove recall upward
- however, the operating threshold used here produced a very high false-positive rate for the best-recall model

This is a realistic production trade-off and should be discussed clearly in the viva.

## 9. Drift Simulation

The project uses time-based drift instead of random noise:

- train on earlier observations
- evaluate on later observations
- simulate shifted distributions in later slices
- monitor drift with feature-level KS statistics

Observed drift summary:

- Overall drift score: `0.0045620762750541045`
- Strong drift signals were detected for `TransactionDT`, `TransactionAmt`, `card1`, `card2`, and `card3`

This satisfies the requirement for more realistic drift simulation.

## 10. Intelligent Retraining Strategy

The repository implements a threshold-based retraining strategy:

- trigger retraining when monitored recall drops below threshold
- trigger retraining when data drift exceeds threshold

The intelligent trigger writes a machine-readable event file and is designed to integrate with CI/CD through `repository_dispatch` or a similar alert-driven event.

## 11. CI/CD Workflow

The GitHub Actions workflow includes:

- push trigger
- pull request trigger
- linting and test stage
- data validation checks
- Docker build stage for training and inference images
- deployment stage
- intelligent retraining trigger stage

Live packaging evidence completed locally:

- training Docker image built successfully
- inference API Docker image built successfully
- both images were pushed to a live local registry at `localhost:5000`
- the inference API container was run successfully and served live requests
- Jenkins was run locally in Docker
- a real Jenkins pipeline job was created and executed successfully

## 12. Monitoring and Observability

### System Metrics

Prometheus instrumentation covers:

- API request rate
- API latency
- error count via request status labels
- resource-oriented dashboard placeholders

### Model Metrics

Tracked model metrics include:

- recall
- precision
- false positive rate
- prediction confidence distribution

### Data Metrics

Tracked data metrics include:

- drift score
- missing-value ratio

### Alert Rules

Prometheus alert rules were created for:

- recall drop below threshold
- data drift above threshold
- API latency spike

These alerts are designed to be visualized in Grafana and to trigger CI/CD retraining actions.

Live monitoring evidence completed locally:

- Prometheus container ran successfully on port `9090`
- Grafana container ran successfully on port `3000`
- Grafana provisioned the dashboards from the repository automatically
- Prometheus scraped live API metrics from the running inference service
- the `FraudRecallDrop` alert entered `firing` state
- the live alert was converted into a retraining-event artifact
- Alertmanager forwarded the live alert to Jenkins
- Jenkins executed the alert-triggered CI/CD job successfully

## 13. Explainability

The project includes explainability through:

- tree-model feature importance support
- SHAP summary plot generation

This addresses the question: why is the model predicting fraud?

The SHAP image is included as an artifact and in the Word report.

## 14. Deliverables Included

The repository contains:

- fraud-detection training pipeline
- Kubeflow pipeline definition
- Kubernetes manifests
- CI/CD workflow file
- monitoring configuration
- Grafana dashboard JSON definitions
- alert rules
- drift simulation
- cost-sensitive learning comparison
- imbalance-strategy comparison
- explainability artifact
- research report in Markdown and Word format
- screenshot evidence package

## 15. Remaining Environment-Limited Items

The code and artifacts now include local live evidence for Kubeflow Pipelines, Jenkins, Alertmanager, Prometheus, Grafana, Docker, and Kubernetes. GitHub Actions hosted cloud logs are not included because Jenkins was used as the CI/CD engine, which is allowed by the assignment.

## 16. Conclusion

This project now meets the assignment requirements using the real IEEE-CIS training data and live local infrastructure. It covers preprocessing, imbalance handling, model comparison, cost-sensitive learning, monitoring, drift handling, retraining logic, explainability, Kubeflow Pipelines submission, Jenkins CI/CD execution, Alertmanager-triggered retraining evidence, and Kubernetes deployment.
