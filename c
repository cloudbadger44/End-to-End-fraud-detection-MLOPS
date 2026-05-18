import { useState } from "react";

  "data/generate_data.py": {
    lang: "python",
    code: `"""
Synthetic fraud dataset generator.
Produces realistic transaction data with ~2% fraud rate.
"""
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random, hashlib, os

fake = Faker()
np.random.seed(42)
random.seed(42)

FRAUD_RATE = 0.02
N_TRANSACTIONS = 500_000
N_MERCHANTS = 5_000
N_CUSTOMERS = 50_000


def _card_hash(cid: int) -> str:
    return hashlib.md5(str(cid).encode()).hexdigest()[:16].upper()


def generate_customers(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": range(n),
        "card_number": [_card_hash(i) for i in range(n)],
        "age": np.random.randint(18, 80, n),
        "credit_limit": np.random.choice(
            [1_000, 5_000, 10_000, 25_000, 50_000], n,
            p=[0.2, 0.35, 0.25, 0.15, 0.05]
        ),
        "signup_date": [
            fake.date_between(start_date="-10y", end_date="today")
            for _ in range(n)
        ],
    })


def generate_merchants(n: int) -> pd.DataFrame:
    categories = [
        "grocery", "restaurant", "gas_station", "retail",
        "online", "travel", "entertainment", "health",
    ]
    return pd.DataFrame({
        "merchant_id": range(n),
        "merchant_name": [fake.company() for _ in range(n)],
        "category": np.random.choice(categories, n),
        "country": np.random.choice(
            ["US", "CA", "GB", "FR", "DE", "AU"],
            n, p=[0.6, 0.1, 0.1, 0.07, 0.07, 0.06]
        ),
        "avg_ticket": np.abs(np.random.normal(45, 30, n)),
    })


def inject_fraud_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic fraud signals into flagged rows."""
    fraud_mask = df["is_fraud"] == 1
    n_fraud = fraud_mask.sum()

    # High-value sudden spike
    df.loc[fraud_mask, "amount"] = np.abs(
        np.random.normal(350, 150, n_fraud)
    )
    # Unusual hours (midnight 00-04)
    df.loc[fraud_mask, "hour"] = np.random.choice(
        list(range(0, 5)) + list(range(22, 24)), n_fraud
    )
    # Often foreign merchant
    df.loc[fraud_mask, "is_foreign"] = np.random.choice(
        [0, 1], n_fraud, p=[0.3, 0.7]
    )
    # Short time since last transaction
    df.loc[fraud_mask, "time_since_last_txn_sec"] = np.random.randint(
        1, 300, n_fraud
    )
    return df


def generate_transactions(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    start = datetime(2023, 1, 1)
    timestamps = [
        start + timedelta(seconds=random.randint(0, 365 * 86_400))
        for _ in range(n)
    ]
    timestamps.sort()

    cust_ids = np.random.randint(0, len(customers), n)
    merch_ids = np.random.randint(0, len(merchants), n)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(n)],
        "customer_id": cust_ids,
        "merchant_id": merch_ids,
        "timestamp": timestamps,
        "amount": np.abs(np.random.lognormal(3.5, 1.2, n)),
        "hour": [t.hour for t in timestamps],
        "day_of_week": [t.weekday() for t in timestamps],
        "is_weekend": [int(t.weekday() >= 5) for t in timestamps],
        "is_foreign": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "device": np.random.choice(
            ["mobile", "web", "pos", "atm"], n,
            p=[0.4, 0.3, 0.25, 0.05]
        ),
        "is_fraud": np.random.choice(
            [0, 1], n, p=[1 - FRAUD_RATE, FRAUD_RATE]
        ),
    })

    # Compute time since last transaction per customer
    df = df.sort_values("timestamp")
    df["prev_txn_time"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["time_since_last_txn_sec"] = (
        (df["timestamp"] - df["prev_txn_time"])
        .dt.total_seconds()
        .fillna(86_400)
        .clip(upper=86_400)
    )
    df = df.drop(columns=["prev_txn_time"])

    return inject_fraud_patterns(df)


def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Generating customers …")
    customers = generate_customers(N_CUSTOMERS)
    customers.to_parquet("data/raw/customers.parquet", index=False)

    print("Generating merchants …")
    merchants = generate_merchants(N_MERCHANTS)
    merchants.to_parquet("data/raw/merchants.parquet", index=False)

    print(f"Generating {N_TRANSACTIONS:,} transactions …")
    txns = generate_transactions(customers, merchants, N_TRANSACTIONS)
    txns.to_parquet("data/raw/transactions.parquet", index=False)

    fraud_pct = txns["is_fraud"].mean() * 100
    print(f"Done. Fraud rate: {fraud_pct:.2f}%")
    print(f"Saved to data/raw/")


if __name__ == "__main__":
    main()
`,
  },

  "features/feature_store.py": {
    lang: "python",
    code: `"""
Feast feature store definitions for fraud detection.
Covers customer-level aggregations + transaction-level signals.
"""
from datetime import timedelta
from feast import (
    Entity, FeatureView, Feature, FileSource,
    ValueType, FeatureStore,
)
from feast.data_format import ParquetFormat
import pandas as pd

# ── Entities ──────────────────────────────────────────────────────────────────

customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Unique customer identifier",
)

# ── Sources ───────────────────────────────────────────────────────────────────

transaction_source = FileSource(
    path="data/processed/transactions_features.parquet",
    event_timestamp_column="timestamp",
    created_timestamp_column="created",
    file_format=ParquetFormat(),
)

customer_stats_source = FileSource(
    path="data/processed/customer_stats.parquet",
    event_timestamp_column="stats_date",
    file_format=ParquetFormat(),
)

# ── Feature Views ─────────────────────────────────────────────────────────────

transaction_features = FeatureView(
    name="transaction_features",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="amount", dtype=ValueType.FLOAT),
        Feature(name="hour", dtype=ValueType.INT32),
        Feature(name="day_of_week", dtype=ValueType.INT32),
        Feature(name="is_weekend", dtype=ValueType.INT32),
        Feature(name="is_foreign", dtype=ValueType.INT32),
        Feature(name="device_encoded", dtype=ValueType.INT32),
        Feature(name="time_since_last_txn_sec", dtype=ValueType.FLOAT),
        Feature(name="log_amount", dtype=ValueType.FLOAT),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "fraud", "version": "v2"},
)

customer_aggregate_features = FeatureView(
    name="customer_aggregate_features",
    entities=["customer_id"],
    ttl=timedelta(days=7),
    features=[
        Feature(name="txn_count_7d", dtype=ValueType.INT64),
        Feature(name="txn_count_30d", dtype=ValueType.INT64),
        Feature(name="avg_amount_7d", dtype=ValueType.FLOAT),
        Feature(name="avg_amount_30d", dtype=ValueType.FLOAT),
        Feature(name="std_amount_30d", dtype=ValueType.FLOAT),
        Feature(name="foreign_txn_ratio_30d", dtype=ValueType.FLOAT),
        Feature(name="unique_merchants_7d", dtype=ValueType.INT64),
        Feature(name="night_txn_ratio_30d", dtype=ValueType.FLOAT),
        Feature(name="max_amount_7d", dtype=ValueType.FLOAT),
        Feature(name="chargeback_count_90d", dtype=ValueType.INT64),
    ],
    online=True,
    source=customer_stats_source,
    tags={"team": "fraud", "version": "v2"},
)


# ── Feature Engineering ───────────────────────────────────────────────────────

DEVICE_MAP = {"pos": 0, "web": 1, "mobile": 2, "atm": 3}


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["device_encoded"] = df["device"].map(DEVICE_MAP).fillna(-1).astype(int)
    df["created"] = pd.Timestamp.now()
    return df


def build_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling window aggregates per customer."""
    import numpy as np
    df = df.sort_values("timestamp")
    results = []

    for cust_id, grp in df.groupby("customer_id"):
        grp = grp.set_index("timestamp").sort_index()
        for cutoff in grp.index:
            w7  = grp.loc[:cutoff].last("7D")
            w30 = grp.loc[:cutoff].last("30D")
            w90 = grp.loc[:cutoff].last("90D")
            results.append({
                "customer_id": cust_id,
                "stats_date": cutoff,
                "txn_count_7d": len(w7),
                "txn_count_30d": len(w30),
                "avg_amount_7d": w7["amount"].mean() if len(w7) else 0.0,
                "avg_amount_30d": w30["amount"].mean() if len(w30) else 0.0,
                "std_amount_30d": w30["amount"].std() if len(w30) > 1 else 0.0,
                "foreign_txn_ratio_30d": (
                    w30["is_foreign"].mean() if len(w30) else 0.0
                ),
                "unique_merchants_7d": w7["merchant_id"].nunique(),
                "night_txn_ratio_30d": (
                    (w30["hour"].between(0, 4)).mean() if len(w30) else 0.0
                ),
                "max_amount_7d": w7["amount"].max() if len(w7) else 0.0,
                "chargeback_count_90d": int(
                    w90.get("is_fraud", pd.Series([0])).sum()
                ),
            })

    return pd.DataFrame(results)


def get_online_features(
    store: FeatureStore,
    customer_ids: list[int],
) -> pd.DataFrame:
    """Retrieve real-time features from online store."""
    entity_rows = [{"customer_id": cid} for cid in customer_ids]
    feature_vector = store.get_online_features(
        features=[
            "transaction_features:amount",
            "transaction_features:hour",
            "transaction_features:is_foreign",
            "transaction_features:time_since_last_txn_sec",
            "transaction_features:log_amount",
            "customer_aggregate_features:txn_count_7d",
            "customer_aggregate_features:avg_amount_30d",
            "customer_aggregate_features:std_amount_30d",
            "customer_aggregate_features:foreign_txn_ratio_30d",
            "customer_aggregate_features:night_txn_ratio_30d",
            "customer_aggregate_features:max_amount_7d",
        ],
        entity_rows=entity_rows,
    )
    return feature_vector.to_df()
`,
  },

  "training/train.py": {
    lang: "python",
    code: `"""
Model training pipeline with MLflow experiment tracking.
Trains XGBoost + LightGBM ensemble; registers best model.
"""
import os, json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud-detection-v2"
MODEL_NAME = "fraud-detector"

FEATURE_COLS = [
    "log_amount", "hour", "day_of_week", "is_weekend", "is_foreign",
    "device_encoded", "time_since_last_txn_sec",
    "txn_count_7d", "txn_count_30d",
    "avg_amount_7d", "avg_amount_30d", "std_amount_30d",
    "foreign_txn_ratio_30d", "unique_merchants_7d",
    "night_txn_ratio_30d", "max_amount_7d", "chargeback_count_90d",
]
TARGET_COL = "is_fraud"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    log.info("Loaded %d rows | fraud=%.2f%%", len(df), y.mean() * 100)
    return X, y


def build_xgb(scale_pos_weight: float) -> Pipeline:
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(model, cv=3, method="isotonic")),
    ])


def build_lgbm(scale_pos_weight: float) -> Pipeline:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(model, cv=3, method="isotonic")),
    ])


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": classification_report(
            y_true, y_pred, output_dict=True
        )["1"]["precision"],
        "recall": classification_report(
            y_true, y_pred, output_dict=True
        )["1"]["recall"],
    }


def compute_shap(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for feature importance."""
    raw_model = pipeline.named_steps["model"].estimator
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(raw_model)
    return explainer.shap_values(X_scaled)


def train(data_path: str, threshold: float = 0.5):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data(data_path)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos
    log.info("Scale pos weight: %.1f", spw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="xgb_lgbm_ensemble") as run:
        mlflow.set_tags({
            "model_type": "ensemble",
            "framework": "xgboost+lightgbm",
            "dataset_rows": len(X),
        })

        # ── Train XGBoost ──────────────────────────────────────────────────
        log.info("Training XGBoost …")
        xgb_pipe = build_xgb(spw)
        xgb_probs = cross_val_predict(
            xgb_pipe, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        xgb_metrics = evaluate(y, xgb_probs, threshold)
        mlflow.log_metrics({f"xgb_{k}": v for k, v in xgb_metrics.items()})
        log.info("XGB  AUC=%.4f  AP=%.4f", xgb_metrics["roc_auc"],
                 xgb_metrics["avg_precision"])

        # ── Train LightGBM ─────────────────────────────────────────────────
        log.info("Training LightGBM …")
        lgb_pipe = build_lgbm(spw)
        lgb_probs = cross_val_predict(
            lgb_pipe, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        lgb_metrics = evaluate(y, lgb_probs, threshold)
        mlflow.log_metrics({f"lgb_{k}": v for k, v in lgb_metrics.items()})
        log.info("LGB  AUC=%.4f  AP=%.4f", lgb_metrics["roc_auc"],
                 lgb_metrics["avg_precision"])

        # ── Ensemble ───────────────────────────────────────────────────────
        ensemble_probs = 0.5 * xgb_probs + 0.5 * lgb_probs
        ens_metrics = evaluate(y, ensemble_probs, threshold)
        mlflow.log_metrics({f"ensemble_{k}": v for k, v in ens_metrics.items()})
        log.info("ENS  AUC=%.4f  AP=%.4f", ens_metrics["roc_auc"],
                 ens_metrics["avg_precision"])

        # ── Fit final models on full data ──────────────────────────────────
        xgb_pipe.fit(X, y)
        lgb_pipe.fit(X, y)

        # ── SHAP feature importance ────────────────────────────────────────
        shap_vals = compute_shap(xgb_pipe, X.sample(min(5000, len(X))))
        importance = dict(zip(FEATURE_COLS, np.abs(shap_vals).mean(axis=0)))
        mlflow.log_dict(importance, "shap_importance.json")
        log.info("Top features: %s",
                 sorted(importance, key=importance.get, reverse=True)[:5])

        # ── Save artifacts ─────────────────────────────────────────────────
        os.makedirs("models", exist_ok=True)
        joblib.dump(xgb_pipe, "models/xgb_pipeline.pkl")
        joblib.dump(lgb_pipe, "models/lgb_pipeline.pkl")

        signature = infer_signature(X, ensemble_probs)
        mlflow.sklearn.log_model(
            xgb_pipe, "xgb_model",
            signature=signature,
            registered_model_name=f"{MODEL_NAME}-xgb",
        )
        mlflow.sklearn.log_model(
            lgb_pipe, "lgb_model",
            signature=signature,
            registered_model_name=f"{MODEL_NAME}-lgb",
        )

        # Save metadata
        meta = {
            "run_id": run.info.run_id,
            "threshold": threshold,
            "feature_cols": FEATURE_COLS,
            "metrics": ens_metrics,
        }
        mlflow.log_dict(meta, "model_metadata.json")
        with open("models/metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info("Run ID: %s", run.info.run_id)
        return run.info.run_id


if __name__ == "__main__":
    train("data/processed/features.parquet")
`,
  },

  "serving/api.py": {
    lang: "python",
    code: `"""
FastAPI inference service for real-time fraud scoring.
Supports single and batch prediction with sub-20ms p99 latency.
"""
from __future__ import annotations
import os, time, json, logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import joblib
import redis
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
import uvicorn

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "models/xgb_pipeline.pkl")
META_PATH    = os.getenv("META_PATH",    "models/metadata.json")
REDIS_URL    = os.getenv("REDIS_URL",    "redis://localhost:6379")
FRAUD_THRESH = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
CACHE_TTL    = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_BATCH    = int(os.getenv("MAX_BATCH_SIZE", "500"))

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("fraud_requests_total",    "Total requests", ["status"])
LATENCY         = Histogram("fraud_latency_seconds", "Inference latency",
                             buckets=[.005,.01,.025,.05,.1,.25,.5,1])
FRAUD_SCORE     = Histogram("fraud_score",           "Fraud probability distribution",
                             buckets=np.linspace(0, 1, 21).tolist())
FRAUD_DETECTED  = Counter("fraud_detected_total",    "Flagged as fraud")
MODEL_VERSION   = Gauge("model_version_info",        "Model version", ["version"])
CACHE_HITS      = Counter("cache_hits_total",        "Redis cache hits")

# ── Global state ──────────────────────────────────────────────────────────────
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model from %s", MODEL_PATH)
    STATE["xgb"] = joblib.load(MODEL_PATH)
    lgb_path = MODEL_PATH.replace("xgb", "lgb")
    STATE["lgb"] = joblib.load(lgb_path) if os.path.exists(lgb_path) else None

    with open(META_PATH) as f:
        meta = json.load(f)
    STATE["meta"] = meta
    STATE["feature_cols"] = meta["feature_cols"]
    STATE["threshold"] = meta.get("threshold", FRAUD_THRESH)
    STATE["version"] = meta.get("run_id", "unknown")[:8]
    MODEL_VERSION.labels(version=STATE["version"]).set(1)

    STATE["redis"] = redis.from_url(REDIS_URL, decode_responses=True)
    log.info("Service ready | version=%s | threshold=%.2f",
             STATE["version"], STATE["threshold"])
    yield
    STATE.clear()


app = FastAPI(
    title="Fraud Detection API",
    version="2.0.0",
    description="Real-time fraud scoring with XGBoost + LightGBM ensemble",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    transaction_id:          str
    log_amount:              float  = Field(..., ge=0, description="log1p(amount)")
    hour:                    int    = Field(..., ge=0, le=23)
    day_of_week:             int    = Field(..., ge=0, le=6)
    is_weekend:              int    = Field(..., ge=0, le=1)
    is_foreign:              int    = Field(..., ge=0, le=1)
    device_encoded:          int    = Field(..., ge=0, le=3)
    time_since_last_txn_sec: float  = Field(..., ge=0)
    txn_count_7d:            int    = Field(0,   ge=0)
    txn_count_30d:           int    = Field(0,   ge=0)
    avg_amount_7d:           float  = Field(0.0, ge=0)
    avg_amount_30d:          float  = Field(0.0, ge=0)
    std_amount_30d:          float  = Field(0.0, ge=0)
    foreign_txn_ratio_30d:   float  = Field(0.0, ge=0, le=1)
    unique_merchants_7d:     int    = Field(0,   ge=0)
    night_txn_ratio_30d:     float  = Field(0.0, ge=0, le=1)
    max_amount_7d:           float  = Field(0.0, ge=0)
    chargeback_count_90d:    int    = Field(0,   ge=0)

    @validator("log_amount")
    def clamp_log_amount(cls, v):
        return min(v, 15.0)


class PredictionResult(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    model_version: str
    cached: bool = False
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[TransactionFeatures]

    @validator("transactions")
    def check_batch_size(cls, v):
        if len(v) > MAX_BATCH:
            raise ValueError(f"Batch exceeds {MAX_BATCH} transactions")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob < 0.2:  return "low"
    if prob < 0.5:  return "medium"
    if prob < 0.8:  return "high"
    return "critical"


def _score_features(feat_dict: dict) -> float:
    import pandas as pd
    row = pd.DataFrame([feat_dict])[STATE["feature_cols"]]
    xgb_prob = STATE["xgb"].predict_proba(row)[0, 1]
    if STATE["lgb"]:
        lgb_prob = STATE["lgb"].predict_proba(row)[0, 1]
        return 0.5 * xgb_prob + 0.5 * lgb_prob
    return xgb_prob


def _cache_key(txn_id: str) -> str:
    return f"fraud:score:{txn_id}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": STATE.get("version")}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResult)
async def predict(txn: TransactionFeatures):
    t0 = time.perf_counter()
    cached = False

    # Check Redis cache
    cache_key = _cache_key(txn.transaction_id)
    cached_val = STATE["redis"].get(cache_key)
    if cached_val:
        CACHE_HITS.inc()
        prob = float(cached_val)
        cached = True
    else:
        feat = txn.dict(exclude={"transaction_id"})
        try:
            prob = float(_score_features(feat))
        except Exception as e:
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))
        STATE["redis"].setex(cache_key, CACHE_TTL, str(prob))

    is_fraud = prob >= STATE["threshold"]
    latency = (time.perf_counter() - t0) * 1000

    LATENCY.observe(latency / 1000)
    FRAUD_SCORE.observe(prob)
    REQUEST_COUNT.labels(status="ok").inc()
    if is_fraud:
        FRAUD_DETECTED.inc()

    return PredictionResult(
        transaction_id=txn.transaction_id,
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        risk_level=_risk_level(prob),
        model_version=STATE["version"],
        cached=cached,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch")
async def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    results = []
    for txn in req.transactions:
        cache_key = _cache_key(txn.transaction_id)
        cached_val = STATE["redis"].get(cache_key)
        if cached_val:
            prob = float(cached_val)
            cached = True
        else:
            feat = txn.dict(exclude={"transaction_id"})
            prob = float(_score_features(feat))
            STATE["redis"].setex(cache_key, CACHE_TTL, str(prob))
            cached = False

        results.append({
            "transaction_id": txn.transaction_id,
            "fraud_probability": round(prob, 6),
            "is_fraud": prob >= STATE["threshold"],
            "risk_level": _risk_level(prob),
            "cached": cached,
        })

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "predictions": results,
        "total_latency_ms": round(total_ms, 2),
        "avg_latency_ms": round(total_ms / len(results), 2),
        "model_version": STATE["version"],
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, workers=4)
`,
  },

  "monitoring/drift_monitor.py": {
    lang: "python",
    code: `"""
Model monitoring: data drift + performance degradation detection.
Generates HTML reports and triggers alerts via webhook/PagerDuty.
"""
import os, json, logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, ClassificationPreset,
    DataQualityPreset,
)
from evidently.metrics import (
    DatasetDriftMetric, DatasetMissingValuesSummaryMetric,
    ColumnDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns, TestShareOfDriftedColumns,
    TestColumnDrift,
)
from scipy import stats
import smtplib
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

ALERT_WEBHOOK   = os.getenv("ALERT_WEBHOOK_URL")
PD_API_KEY      = os.getenv("PAGERDUTY_API_KEY")
EMAIL_FROM      = os.getenv("ALERT_EMAIL_FROM")
EMAIL_TO        = os.getenv("ALERT_EMAIL_TO")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.1"))
PSI_THRESHOLD   = float(os.getenv("PSI_THRESHOLD", "0.2"))
REPORT_DIR      = os.getenv("REPORT_DIR", "monitoring/reports")

CRITICAL_FEATURES = [
    "log_amount", "time_since_last_txn_sec",
    "foreign_txn_ratio_30d", "avg_amount_30d",
]


# ── PSI (Population Stability Index) ─────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI < 0.1: no change | 0.1-0.2: slight | >0.2: significant."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ── KS Test ───────────────────────────────────────────────────────────────────

def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    stat, p_value = stats.ks_2samp(reference, current)
    return {"statistic": stat, "p_value": p_value, "drifted": p_value < 0.05}


# ── Evidently Reports ─────────────────────────────────────────────────────────

def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str,
) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Full data drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetMissingValuesSummaryMetric(),
    ])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    html_path = f"{REPORT_DIR}/{report_name}_drift.html"
    drift_report.save_html(html_path)

    result = drift_report.as_dict()
    drift_metrics = result["metrics"][0]["result"]

    summary = {
        "report_path": html_path,
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_drift": drift_metrics.get("dataset_drift", False),
        "drift_share": drift_metrics.get("share_of_drifted_columns", 0.0),
        "n_drifted_cols": drift_metrics.get("number_of_drifted_columns", 0),
        "feature_psi": {},
    }

    # Per-feature PSI
    for feat in CRITICAL_FEATURES:
        if feat in reference_df.columns and feat in current_df.columns:
            psi = compute_psi(
                reference_df[feat].dropna().values,
                current_df[feat].dropna().values,
            )
            summary["feature_psi"][feat] = round(psi, 4)

    return summary


def generate_performance_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str,
) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)

    perf_report = Report(metrics=[ClassificationPreset()])
    perf_report.run(reference_data=reference_df, current_data=current_df)
    html_path = f"{REPORT_DIR}/{report_name}_performance.html"
    perf_report.save_html(html_path)

    result   = perf_report.as_dict()["metrics"][0]["result"]
    cur_roc  = result.get("current", {}).get("roc_auc", None)
    ref_roc  = result.get("reference", {}).get("roc_auc", None)

    return {
        "report_path": html_path,
        "current_roc_auc": cur_roc,
        "reference_roc_auc": ref_roc,
        "degradation": (ref_roc - cur_roc) if (cur_roc and ref_roc) else None,
    }


# ── Alerting ──────────────────────────────────────────────────────────────────

def _send_slack_alert(message: str):
    if not ALERT_WEBHOOK:
        log.warning("No webhook configured, skipping Slack alert.")
        return
    payload = {
        "text": f":rotating_light: *Fraud Model Alert*\n{message}",
        "username": "MLOps Monitor",
        "icon_emoji": ":bar_chart:",
    }
    try:
        r = requests.post(ALERT_WEBHOOK, json=payload, timeout=5)
        r.raise_for_status()
    except Exception as e:
        log.error("Failed to send Slack alert: %s", e)


def _page_on_call(summary: str, severity: str = "critical"):
    if not PD_API_KEY:
        return
    payload = {
        "routing_key": PD_API_KEY,
        "event_action": "trigger",
        "payload": {
            "summary": f"Fraud Model: {summary}",
            "severity": severity,
            "source": "mlops-monitor",
            "timestamp": datetime.utcnow().isoformat(),
        },
    }
    try:
        r = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload, timeout=5,
        )
        r.raise_for_status()
    except Exception as e:
        log.error("PagerDuty alert failed: %s", e)


def check_and_alert(drift_summary: dict, perf_summary: dict):
    alerts = []

    if drift_summary.get("dataset_drift"):
        alerts.append(
            f"Dataset drift detected ({drift_summary['n_drifted_cols']} cols)"
        )

    high_psi = {
        feat: psi
        for feat, psi in drift_summary.get("feature_psi", {}).items()
        if psi > PSI_THRESHOLD
    }
    if high_psi:
        alerts.append(f"High PSI: {high_psi}")

    degradation = perf_summary.get("degradation")
    if degradation and degradation > 0.05:
        alerts.append(
            f"ROC-AUC degraded by {degradation:.3f} "
            f"({perf_summary['reference_roc_auc']:.3f} → "
            f"{perf_summary['current_roc_auc']:.3f})"
        )

    if alerts:
        message = "\n".join(f"• {a}" for a in alerts)
        log.warning("ALERTS:\n%s", message)
        _send_slack_alert(message)
        if degradation and degradation > 0.1:
            _page_on_call(alerts[0], severity="critical")
    else:
        log.info("All metrics healthy — no alerts triggered.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_monitoring_cycle(
    reference_path: str,
    current_path: str,
    report_tag: Optional[str] = None,
):
    tag = report_tag or datetime.utcnow().strftime("%Y%m%d_%H%M")
    log.info("Loading reference data from %s", reference_path)
    ref_df = pd.read_parquet(reference_path)
    log.info("Loading current data from %s", current_path)
    cur_df = pd.read_parquet(current_path)

    drift_summary = generate_drift_report(ref_df, cur_df, tag)
    perf_summary  = generate_performance_report(ref_df, cur_df, tag)
    check_and_alert(drift_summary, perf_summary)

    combined = {**drift_summary, **perf_summary, "tag": tag}
    out_path = f"{REPORT_DIR}/{tag}_summary.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info("Summary saved to %s", out_path)
    return combined


if __name__ == "__main__":
    run_monitoring_cycle(
        reference_path="data/processed/reference_set.parquet",
        current_path="data/processed/current_window.parquet",
    )
`,
  },

  "pipelines/training_dag.py": {
    lang: "python",
    code: `"""
Airflow DAG: end-to-end fraud model training pipeline.
Schedule: weekly on Sunday 02:00 UTC.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.trigger_rule import TriggerRule
import mlflow

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ml-alerts@company.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

MODEL_QUALITY_THRESHOLD = {
    "roc_auc": 0.92,
    "avg_precision": 0.75,
}


def _ingest_data(**ctx):
    import subprocess
    result = subprocess.run(
        ["python", "data/generate_data.py"],
        capture_output=True, text=True, check=True,
    )
    print(result.stdout)
    ctx["ti"].xcom_push(key="data_path",
                        value="data/processed/features.parquet")


def _feature_engineering(**ctx):
    import pandas as pd
    from features.feature_store import (
        build_transaction_features,
        build_customer_aggregates,
    )
    df = pd.read_parquet("data/raw/transactions.parquet")
    df = build_transaction_features(df)
    agg = build_customer_aggregates(df)
    merged = df.merge(agg, on=["customer_id", "timestamp"],
                      how="left").fillna(0)
    merged.to_parquet("data/processed/features.parquet", index=False)
    print(f"Feature engineering complete: {len(merged):,} rows")


def _train_model(**ctx):
    from training.train import train
    run_id = train("data/processed/features.parquet")
    ctx["ti"].xcom_push(key="run_id", value=run_id)
    print(f"Training complete — MLflow run: {run_id}")


def _evaluate_model(**ctx):
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train_model")
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    auc = metrics.get("ensemble_roc_auc", 0)
    ap  = metrics.get("ensemble_avg_precision", 0)
    print(f"Evaluation — AUC: {auc:.4f}, AP: {ap:.4f}")

    passed = (
        auc >= MODEL_QUALITY_THRESHOLD["roc_auc"]
        and ap  >= MODEL_QUALITY_THRESHOLD["avg_precision"]
    )
    ctx["ti"].xcom_push(key="passed", value=passed)
    ctx["ti"].xcom_push(key="auc",    value=round(auc, 4))
    ctx["ti"].xcom_push(key="ap",     value=round(ap, 4))
    return "promote_model" if passed else "reject_model"


def _promote_model(**ctx):
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train_model")
    client = mlflow.MlflowClient()

    # Archive current Production model
    for mv in client.search_model_versions("name='fraud-detector-xgb'"):
        if mv.current_stage == "Production":
            client.transition_model_version_stage(
                name="fraud-detector-xgb",
                version=mv.version,
                stage="Archived",
            )

    # Promote new version
    latest = client.search_model_versions(
        f"name='fraud-detector-xgb' and run_id='{run_id}'"
    )[0]
    client.transition_model_version_stage(
        name="fraud-detector-xgb",
        version=latest.version,
        stage="Production",
    )
    print(f"Promoted version {latest.version} to Production")


def _reject_model(**ctx):
    auc = ctx["ti"].xcom_pull(key="auc", task_ids="evaluate_model")
    ap  = ctx["ti"].xcom_pull(key="ap",  task_ids="evaluate_model")
    raise ValueError(
        f"Model rejected: AUC={auc} (min={MODEL_QUALITY_THRESHOLD['roc_auc']}), "
        f"AP={ap} (min={MODEL_QUALITY_THRESHOLD['avg_precision']})"
    )


def _run_monitoring(**ctx):
    from monitoring.drift_monitor import run_monitoring_cycle
    run_monitoring_cycle(
        reference_path="data/processed/reference_set.parquet",
        current_path="data/processed/features.parquet",
    )


with DAG(
    dag_id="fraud_detection_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Weekly fraud model retraining pipeline",
    schedule_interval="0 2 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["fraud", "mlops", "production"],
    doc_md="""
    ## Fraud Detection Training Pipeline
    Runs every Sunday at 02:00 UTC.
    Steps: ingest → features → train → evaluate → promote/reject → monitor.
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest_data,
    )

    features = PythonOperator(
        task_id="feature_engineering",
        python_callable=_feature_engineering,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        execution_timeout=timedelta(hours=4),
    )

    evaluate = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=_reject_model,
    )

    monitor = PythonOperator(
        task_id="run_monitoring",
        python_callable=_run_monitoring,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    notify_success = SlackWebhookOperator(
        task_id="notify_success",
        slack_webhook_conn_id="slack_mlops",
        message=(
            ":white_check_mark: Fraud model pipeline succeeded!\n"
            "AUC={{ ti.xcom_pull(key='auc', task_ids='evaluate_model') }}"
        ),
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    notify_failure = SlackWebhookOperator(
        task_id="notify_failure",
        slack_webhook_conn_id="slack_mlops",
        message=":x: Fraud model pipeline FAILED! Check Airflow logs.",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    (
        start
        >> ingest
        >> features
        >> train
        >> evaluate
        >> [promote, reject]
        >> monitor
        >> [notify_success, notify_failure]
        >> end
    )
`,
  },

  ".github/workflows/ci_cd.yml": {
    lang: "yaml",
    code: `name: Fraud MLOps CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      force_retrain:
        description: Force model retraining
        type: boolean
        default: false

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_NAME: \${{ github.repository }}/fraud-api
  MLFLOW_TRACKING_URI: \${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  # ── Lint & Type Check ──────────────────────────────────────────────────────
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: \${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install ruff mypy types-requests
      - run: ruff check .
      - run: mypy training/ serving/ monitoring/ --ignore-missing-imports

  # ── Unit Tests ─────────────────────────────────────────────────────────────
  test:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: lint
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: \${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements-dev.txt
      - name: Run tests with coverage
        run: |
          pytest tests/ \\
            --cov=. \\
            --cov-report=xml \\
            --cov-report=html \\
            --junitxml=reports/junit.xml \\
            -v
        env:
          REDIS_URL: redis://localhost:6379
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
          token: \${{ secrets.CODECOV_TOKEN }}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-reports
          path: reports/

  # ── Data Validation ────────────────────────────────────────────────────────
  data-validation:
    name: Data Validation
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: \${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt great-expectations
      - name: Run Great Expectations
        run: python -m pytest tests/data/ -v -k "data"

  # ── Train & Register Model ─────────────────────────────────────────────────
  train:
    name: Train Model
    runs-on: ubuntu-latest
    needs: [test, data-validation]
    if: |
      github.ref == 'refs/heads/main' ||
      github.event.inputs.force_retrain == 'true'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: \${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - name: Generate training data
        run: python data/generate_data.py
      - name: Run feature engineering
        run: python features/build_features.py
      - name: Train models
        run: python training/train.py
        env:
          MLFLOW_TRACKING_URI: \${{ env.MLFLOW_TRACKING_URI }}
          MLFLOW_EXPERIMENT_NAME: fraud-detection-ci
      - name: Export model artifacts
        run: python training/export_model.py --format onnx
      - uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: models/
          retention-days: 30

  # ── Build & Push Docker Image ──────────────────────────────────────────────
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: train
    permissions:
      contents: read
      packages: write
    outputs:
      image-digest: \${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: model-artifacts
          path: models/
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: \${{ env.REGISTRY }}
          username: \${{ github.actor }}
          password: \${{ secrets.GITHUB_TOKEN }}
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: \${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=sha-
            type=ref,event=branch
            type=semver,pattern={{version}}
            latest
      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: infra/Dockerfile
          push: true
          tags: \${{ steps.meta.outputs.tags }}
          labels: \${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

  # ── Security Scan ──────────────────────────────────────────────────────────
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: >-
            \${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}:latest
          format: sarif
          output: trivy-results.sarif
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

  # ── Deploy to Staging ──────────────────────────────────────────────────────
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build, security]
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-kubectl@v4
      - uses: azure/k8s-set-context@v4
        with:
          kubeconfig: \${{ secrets.KUBE_CONFIG_STAGING }}
      - name: Deploy
        run: |
          kubectl set image deployment/fraud-api \\
            fraud-api=\${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}@\${{ needs.build.outputs.image-digest }} \\
            -n fraud-staging
          kubectl rollout status deployment/fraud-api -n fraud-staging --timeout=5m
      - name: Smoke test
        run: |
          ENDPOINT=\${{ secrets.STAGING_ENDPOINT }}
          curl -sf \$ENDPOINT/health
          python tests/smoke_test.py --endpoint \$ENDPOINT

  # ── Deploy to Production ───────────────────────────────────────────────────
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment:
      name: production
      url: https://fraud-api.company.com
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-kubectl@v4
      - uses: azure/k8s-set-context@v4
        with:
          kubeconfig: \${{ secrets.KUBE_CONFIG_PROD }}
      - name: Canary deploy (10%)
        run: |
          kubectl set image deployment/fraud-api-canary \\
            fraud-api=\${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}@\${{ needs.build.outputs.image-digest }} \\
            -n fraud-production
          kubectl rollout status deployment/fraud-api-canary -n fraud-production --timeout=5m
      - name: Wait for canary validation (15 min)
        run: sleep 900
      - name: Promote to full rollout
        run: |
          kubectl set image deployment/fraud-api \\
            fraud-api=\${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}@\${{ needs.build.outputs.image-digest }} \\
            -n fraud-production
          kubectl rollout status deployment/fraud-api -n fraud-production --timeout=10m
      - name: Notify deployment
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {"text": ":rocket: Fraud API deployed to production!\nSHA: \${{ github.sha }}"}
        env:
          SLACK_WEBHOOK_URL: \${{ secrets.SLACK_WEBHOOK_URL }}
`,
  },

  "infra/Dockerfile": {
    lang: "dockerfile",
    code: `# ── Builder stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential gcc libgomp1 && \\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 curl && \\
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /home/mlops/.local
ENV PATH=/home/mlops/.local/bin:$PATH

COPY serving/     ./serving/
COPY models/      ./models/
COPY features/    ./features/

RUN chown -R mlops:mlops /app
USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \\
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.api:app", \\
     "--host", "0.0.0.0", "--port", "8000", \\
     "--workers", "4", "--log-level", "info"]
`,
  },

  "infra/k8s-deployment.yaml": {
    lang: "yaml",
    code: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
  namespace: fraud-production
  labels:
    app: fraud-api
    version: v2
    team: ml-platform
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: /metrics
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: fraud-api
  template:
    metadata:
      labels:
        app: fraud-api
        version: v2
    spec:
      serviceAccountName: fraud-api-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: fraud-api
          image: ghcr.io/your-org/fraud-detection-mlops/fraud-api:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: MODEL_PATH
              value: /app/models/xgb_pipeline.pkl
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: fraud-api-secrets
                  key: redis-url
            - name: FRAUD_THRESHOLD
              valueFrom:
                configMapKeyRef:
                  name: fraud-api-config
                  key: fraud-threshold
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                secretKeyRef:
                  name: fraud-api-secrets
                  key: mlflow-uri
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 2000m
              memory: 2Gi
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          volumeMounts:
            - name: model-store
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: model-store
          persistentVolumeClaim:
            claimName: fraud-model-pvc
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: fraud-api

---
apiVersion: v1
kind: Service
metadata:
  name: fraud-api-svc
  namespace: fraud-production
spec:
  selector:
    app: fraud-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-api-hpa
  namespace: fraud-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: fraud_requests_per_second
        target:
          type: AverageValue
          averageValue: "500"

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: fraud-api-pdb
  namespace: fraud-production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: fraud-api
`,
  },

  "tests/test_api.py": {
    lang: "python",
    code: `"""
Integration tests for the fraud detection API.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import joblib

# Mock dependencies before importing app
with patch("redis.from_url") as mock_redis, \\
     patch("joblib.load") as mock_joblib:

    mock_redis.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
    mock_joblib.return_value = mock_model

    from serving.api import app, STATE

    STATE["xgb"] = mock_model
    STATE["lgb"] = None
    STATE["meta"] = {"run_id": "abc12345", "threshold": 0.5,
                     "feature_cols": list(range(17))}
    from features.feature_store import FEATURE_COLS as FC
    STATE["feature_cols"] = FC
    STATE["threshold"] = 0.5
    STATE["version"] = "abc12345"
    STATE["redis"] = MagicMock()
    STATE["redis"].get.return_value = None

client = TestClient(app)

SAMPLE_TXN = {
    "transaction_id": "TXN00000001",
    "log_amount": 4.5,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_foreign": 0,
    "device_encoded": 1,
    "time_since_last_txn_sec": 3600.0,
    "txn_count_7d": 5,
    "txn_count_30d": 18,
    "avg_amount_7d": 45.0,
    "avg_amount_30d": 42.0,
    "std_amount_30d": 12.5,
    "foreign_txn_ratio_30d": 0.05,
    "unique_merchants_7d": 3,
    "night_txn_ratio_30d": 0.02,
    "max_amount_7d": 95.0,
    "chargeback_count_90d": 0,
}


class TestHealthEndpoint:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=SAMPLE_TXN)
        assert r.status_code == 200

    def test_predict_schema(self):
        r = client.post("/predict", json=SAMPLE_TXN)
        data = r.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "latency_ms" in data

    def test_fraud_probability_in_range(self):
        r = client.post("/predict", json=SAMPLE_TXN)
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_levels(self):
        for mock_prob, expected_level in [
            (0.05, "low"),
            (0.35, "medium"),
            (0.65, "high"),
            (0.95, "critical"),
        ]:
            STATE["redis"].get.return_value = str(mock_prob)
            r = client.post("/predict", json=SAMPLE_TXN)
            assert r.json()["risk_level"] == expected_level

    def test_missing_required_field(self):
        bad = {k: v for k, v in SAMPLE_TXN.items() if k != "log_amount"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_hour(self):
        bad = {**SAMPLE_TXN, "hour": 99}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_cache_hit(self):
        STATE["redis"].get.return_value = "0.03"
        r = client.post("/predict", json=SAMPLE_TXN)
        assert r.json()["cached"] is True


class TestBatchEndpoint:
    def test_batch_predict(self):
        STATE["redis"].get.return_value = None
        payload = {"transactions": [SAMPLE_TXN] * 3}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 3
        assert "avg_latency_ms" in data

    def test_batch_too_large(self):
        payload = {"transactions": [SAMPLE_TXN] * 501}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert b"fraud_requests_total" in r.content
`,
  },
};

const FILE_TREE = [
  { name: "README.md",               icon: "ti-file-description", color: "#5DCAA5" },
  { name: "data/generate_data.py",   icon: "ti-database",          color: "#7F77DD" },
  { name: "features/feature_store.py", icon: "ti-stack",           color: "#7F77DD" },
  { name: "training/train.py",       icon: "ti-brain",             color: "#378ADD" },
  { name: "serving/api.py",          icon: "ti-api",               color: "#378ADD" },
  { name: "monitoring/drift_monitor.py", icon: "ti-activity",      color: "#EF9F27" },
  { name: "pipelines/training_dag.py", icon: "ti-git-branch",      color: "#EF9F27" },
  { name: ".github/workflows/ci_cd.yml", icon: "ti-brand-github",  color: "#D85A30" },
  { name: "infra/Dockerfile",        icon: "ti-brand-docker",      color: "#185FA5" },
  { name: "infra/k8s-deployment.yaml", icon: "ti-topology-star",   color: "#185FA5" },
  { name: "tests/test_api.py",       icon: "ti-test-pipe",         color: "#639922" },
];

function highlight(code, lang) {
  const escape = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  if (lang === "markdown") return `<span style="color:var(--color-text-primary)">${escape(code)}</span>`;

  const rules = {
    python: [
      [/(#[^\n]*)/g, "comment"],
      [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|"""[\s\S]*?"""|'''[\s\S]*?''')/g, "string"],
      [/\b(def|class|import|from|return|if|elif|else|for|while|try|except|with|as|in|not|and|or|is|None|True|False|lambda|yield|raise|pass|break|continue|async|await)\b/g, "keyword"],
      [/\b(int|str|float|bool|list|dict|tuple|set|Optional|Union|Any|pd|np|os|json|logging)\b/g, "type"],
      [/\b(\d+\.?\d*)\b/g, "number"],
    ],
    yaml: [
      [/(#[^\n]*)/g, "comment"],
      [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, "string"],
      [/^(\s*[\w-]+):/gm, "key"],
      [/\b(true|false|null)\b/g, "keyword"],
      [/\b(\d+\.?\d*)\b/g, "number"],
    ],
    dockerfile: [
      [/(#[^\n]*)/g, "comment"],
      [/^(FROM|RUN|COPY|WORKDIR|ENV|EXPOSE|CMD|ENTRYPOINT|USER|HEALTHCHECK|ARG|LABEL|ADD|VOLUME|STOPSIGNAL|ONBUILD)\b/gm, "keyword"],
      [/("(?:[^"\\]|\\.)*")/g, "string"],
    ],
  };

  const colors = {
    comment: "#9ca3af",
    string: "#5DCAA5",
    keyword: "#378ADD",
    type: "#EF9F27",
    number: "#D85A30",
    key: "#7F77DD",
  };

  let result = escape(code);
  const ruleSet = rules[lang] || rules.python;

  // Simple sequential replacement with placeholders
  const matches = [];
  for (const [regex, type] of ruleSet) {
    let m;
    const r = new RegExp(regex.source, regex.flags);
    while ((m = r.exec(escape(code))) !== null) {
      matches.push({ start: m.index, end: m.index + m[0].length, text: m[0], type });
    }
  }

  matches.sort((a, b) => a.start - b.start);
  let out = "";
  let pos = 0;
  const src = escape(code);
  for (const { start, end, text, type } of matches) {
    if (start < pos) continue;
    out += src.slice(pos, start);
    out += `<span style="color:${colors[type] || "inherit"}">${text}</span>`;
    pos = end;
  }
  out += src.slice(pos);
  return out;
}

export default function FraudMLOpsExplorer() {
  const [activeFile, setActiveFile] = useState("README.md");
  const [copied, setCopied] = useState(false);

  const file = FILES[activeFile];

  function copyCode() {
    navigator.clipboard.writeText(file.code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  const highlighted = file ? highlight(file.code, file.lang) : "";

  return (
    <div style={{ display: "flex", height: 580, border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", overflow: "hidden", fontFamily: "var(--font-mono)", fontSize: 13 }}>

      {/* Sidebar */}
      <div style={{ width: 230, minWidth: 230, background: "var(--color-background-secondary)", borderRight: "0.5px solid var(--color-border-tertiary)", overflowY: "auto", display: "flex", flexDirection: "column" }}>
        <div style={{ padding: "12px 14px 8px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
          <p style={{ margin: 0, fontSize: 11, fontWeight: 500, letterSpacing: "0.05em", color: "var(--color-text-secondary)", fontFamily: "var(--font-sans)", textTransform: "uppercase" }}>
            fraud-detection-mlops
          </p>
        </div>
        {FILE_TREE.map(({ name, icon, color }) => (
          <button
            key={name}
            onClick={() => setActiveFile(name)}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "7px 14px",
              background: activeFile === name ? "var(--color-background-primary)" : "transparent",
              border: "none",
              borderLeft: activeFile === name ? `2px solid ${color}` : "2px solid transparent",
              cursor: "pointer", textAlign: "left", width: "100%",
              color: activeFile === name ? "var(--color-text-primary)" : "var(--color-text-secondary)",
              fontSize: 12,
            }}
          >
            <i className={`ti ${icon}`} style={{ fontSize: 14, color, flexShrink: 0 }} aria-hidden="true" />
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
          </button>
        ))}
      </div>

      {/* Code pane */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Topbar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 16px", borderBottom: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)" }}>
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{activeFile}</span>
          <div style={{ display: "flex", gap: 8 }}>
            <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", border: "0.5px solid var(--color-border-tertiary)" }}>
              {file?.lang}
            </span>
            <button
              onClick={copyCode}
              style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, padding: "2px 10px", borderRadius: 4, background: "transparent", color: "var(--color-text-secondary)", border: "0.5px solid var(--color-border-secondary)", cursor: "pointer" }}
            >
              <i className={`ti ${copied ? "ti-check" : "ti-copy"}`} style={{ fontSize: 12 }} aria-hidden="true" />
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>

        {/* Code */}
        <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", background: "var(--color-background-primary)" }}>
          <pre style={{ margin: 0, fontSize: 12, lineHeight: 1.65, whiteSpace: "pre", color: "var(--color-text-primary)" }}>
            <code dangerouslySetInnerHTML={{ __html: highlighted }} />
          </pre>
        </div>

        {/* Footer stats */}
        <div style={{ padding: "6px 16px", borderTop: "0.5px solid var(--color-border-tertiary)", display: "flex", gap: 20, background: "var(--color-background-secondary)" }}>
          {[
            ["Lines", file?.code.split("\n").length],
            ["Chars", file?.code.length.toLocaleString()],
            ["Lang", file?.lang],
          ].map(([label, val]) => (
            <span key={label} style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>
              {label}: <strong style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{val}</strong>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
