# Runbook

## 1. Compile Kubeflow Pipeline

```powershell
python kubeflow/pipeline.py
```

## 2. Train Locally

```powershell
python -m fraud_mlops.train --config configs/training_config.yaml
```

## 3. Start API

```powershell
uvicorn fraud_mlops.inference_api:app --host 0.0.0.0 --port 8000
```

## 4. Apply Kubernetes Resources

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/resource-quota.yaml
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/persistent-volume-claim.yaml
kubectl apply -f k8s/inference-api-deployment.yaml
```

## 5. Monitoring Setup

Deploy Prometheus with:

```powershell
kubectl create configmap prometheus-config --from-file=monitoring/prometheus.yml -n fraud-mlops
kubectl create configmap prometheus-alerts --from-file=monitoring/alert_rules.yml -n fraud-mlops
```

Import the Grafana dashboards from `monitoring/grafana/dashboards/`.
