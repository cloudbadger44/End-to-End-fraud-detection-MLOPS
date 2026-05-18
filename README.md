End-to-end fraud detection system with full MLOps lifecycle:
feature engineering, model training, real-time inference, drift
monitoring, and CI/CD automation.

## Architecture
```
Raw Data → Feature Pipeline → Training → Registry
                                             ↓
Alert ← Monitor ← Serving API ← Promoted Model
```

## Stack
| Layer | Tool |
|---|---|
| Orchestration | Apache Airflow |
| Experiment Tracking | MLflow |
| Feature Store | Feast |
| Serving | FastAPI + BentoML |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions |
| Container | Docker + Kubernetes |
| Data | PostgreSQL + Redis |
