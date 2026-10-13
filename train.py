"""Leakage-aware PaySim benchmark and pre-authorization model training.

Usage: python train.py [--data path/to/PaySim.csv] [--sample 1000000]
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import (POST_AUTH_NUM_COLS, POST_AUTH_RAW_COLS, PRE_AUTH_NUM_COLS,
                      PRE_AUTH_RAW_COLS, TX_TYPES, add_post_auth_features,
                      add_pre_auth_features)

MODELS_DIR = Path("models")
SEED = 42
XGB_RF_SELECTION_TOLERANCE = 0.01


def build_pipeline(model, regime: str, scale: bool = False):
    num_cols = PRE_AUTH_NUM_COLS if regime == "pre_authorization" else POST_AUTH_NUM_COLS
    feature_fn = add_pre_auth_features if regime == "pre_authorization" else add_post_auth_features
    prep = ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", num_cols),
         ("cat", OneHotEncoder(categories=[TX_TYPES], handle_unknown="ignore"), ["type"])],
        sparse_threshold=0,
    )
    return Pipeline([("features", FunctionTransformer(feature_fn)),
                     ("prep", prep), ("model", model)])


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def candidate_models(y_train):
    positives = max(int((y_train == 1).sum()), 1)
    pos_weight = int((y_train == 0).sum()) / positives
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=SEED), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=14, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), False),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=100, max_depth=14, class_weight="balanced", n_jobs=-1, random_state=SEED), False),
        "XGBoost": (XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=pos_weight, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED), False),
    }


def run_regime(name, raw_cols, X_train, y_train, X_val, y_val, X_test, y_test):
    validation, test, fitted, training_seconds = {}, {}, {}, {}
    source_cols = PRE_AUTH_RAW_COLS if name == "pre_authorization" else POST_AUTH_RAW_COLS
    train_x, val_x, test_x = X_train[source_cols], X_val[source_cols], X_test[source_cols]
    for model_name, (estimator, scale) in candidate_models(y_train).items():
        pipe = build_pipeline(estimator, name, scale)
        started = time.perf_counter()
        pipe.fit(train_x, y_train)
        elapsed = time.perf_counter() - started
        validation[model_name] = evaluate(pipe, val_x, y_val)
        test[model_name] = evaluate(pipe, test_x, y_test)
        training_seconds[model_name] = elapsed
        test[model_name]["train_seconds"] = round(elapsed, 2)
        fitted[model_name] = pipe
        print(f"{name:18} {model_name:20} validation PR-AUC={validation[model_name]['pr_auc']:.4f}")
    best = max(validation, key=lambda model_name: validation[model_name]["pr_auc"])
    selection_rule = "highest validation PR-AUC"
    xgb_name, forest_name = "XGBoost", "Random Forest"
    if best == forest_name and xgb_name in validation:
        score_gap = validation[forest_name]["pr_auc"] - validation[xgb_name]["pr_auc"]
        if (score_gap <= XGB_RF_SELECTION_TOLERANCE
                and training_seconds[xgb_name] < training_seconds[forest_name]):
            best = xgb_name
            selection_rule = (
                "XGBoost preferred over Random Forest within "
                f"{XGB_RF_SELECTION_TOLERANCE:.2f} validation PR-AUC when faster"
            )
    return {
        "best_model": best,
        "validation": validation,
        "test": test,
        "selection_rule": selection_rule,
    }, fitted[best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--sample", type=int, default=1_000_000, help="rows to use; 0 means all rows")
    args = parser.parse_args()
    raw_cols = list(dict.fromkeys(PRE_AUTH_RAW_COLS + POST_AUTH_RAW_COLS))
    print("Loading data...")
    df = pd.read_csv(args.data, usecols=raw_cols + ["isFraud"])
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=SEED)
    print(f"Rows: {len(df):,} | Fraud: {int(df.isFraud.sum()):,} ({df.isFraud.mean():.4%})")

    X, y = df[raw_cols], df["isFraud"].astype(int)
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=SEED)
    print(f"Split: train={len(X_train):,}, validation={len(X_val):,}, test={len(X_test):,}")

    # Both feature regimes share identical, stratified transaction partitions.
    pre_results, pre_model = run_regime("pre_authorization", PRE_AUTH_RAW_COLS, X_train, y_train, X_val, y_val, X_test, y_test)
    post_results, _ = run_regime("post_authorization", POST_AUTH_RAW_COLS, X_train, y_train, X_val, y_val, X_test, y_test)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pre_model, MODELS_DIR / "best_model.joblib")
    metrics = {
        "protocol": "stratified 70/15/15; random_state=42; fixed threshold=0.5",
        "best_model": pre_results["best_model"],
        "active_feature_regime": "pre_authorization",
        "rows_used": int(len(df)), "fraud_rate": float(y.mean()),
        "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "results": pre_results["test"],
        "regimes": {"pre_authorization": pre_results, "post_authorization": post_results},
        "selection_metric": "validation PR-AUC (average precision)",
        "selection_policy": (
            "Choose the highest validation PR-AUC; if Random Forest leads XGBoost "
            f"by no more than {XGB_RF_SELECTION_TOLERANCE:.2f} and XGBoost trains faster, prefer XGBoost."
        ),
        "test_set_used_for_selection": False,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Selected pre-authorization model: {pre_results['best_model']}")
    print("Saved models/best_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
