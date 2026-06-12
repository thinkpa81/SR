
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from pipeline_lib.modeling import (
    HAS_CATBOOST,
    HAS_XGBOOST,
    ensure_dir,
    fit_sklearn_pipeline,
    load_analysis_base,
    predict_catboost,
    sanitize_model_input,
    select_model_columns,
    setup_logging,
)
from pipeline_lib.paths import resolve_project_dir

if HAS_CATBOOST:
    from catboost import CatBoostClassifier

if HAS_XGBOOST:
    from xgboost import XGBClassifier


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

                                                                                    
INNER_TRAIN_WAVES = "1-17"
INNER_VALIDATION_WAVES = "18-20"
INNER_TRAIN_END_WAVE = 17
INNER_VALID_START_WAVE = 18
INNER_VALID_END_WAVE = 20

TARGET_COL = "exit_label_t1"
RANDOM_STATE = 42
MISSING_CATEGORY = "__MISSING__"
TOP_SHARE = 0.20                                               


def catboost_configs() -> dict[str, dict]:
    return {
        "cat_1": {"iterations": 200, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        "cat_2": {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        "cat_3": {"iterations": 400, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5.0},
    }


def xgboost_configs() -> dict[str, dict]:
    return {
        "xgb_1": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        "xgb_2": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        "xgb_3": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9},
    }


def is_primary_fixed_config(config_id: str) -> bool:
    return config_id in {"cat_2", "xgb_2"}


def recall_at_share(y_true: np.ndarray, y_prob: np.ndarray, share: float) -> float:
    top_n = max(1, int(np.ceil(len(y_true) * share)))
    selected = np.argsort(-y_prob)[:top_n]
    positives = y_true.sum()
    return float(y_true[selected].sum() / positives) if positives else np.nan


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    recall_20 = recall_at_share(y_true, y_prob, TOP_SHARE)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "recall_at_20": recall_20,
        "lift_at_20": float(recall_20 / TOP_SHARE) if np.isfinite(recall_20) else np.nan,
    }


def prepare_data(df: pd.DataFrame):
    train = df[df["wave"] <= INNER_TRAIN_END_WAVE].copy()
    valid = df[(df["wave"] >= INNER_VALID_START_WAVE) & (df["wave"] <= INNER_VALID_END_WAVE)].copy()
    feature_cols, numeric_features, categorical_features = select_model_columns(train)
    X_train = sanitize_model_input(train[feature_cols], numeric_features, categorical_features)
    X_valid = sanitize_model_input(valid[feature_cols], numeric_features, categorical_features)
    y_train = train[TARGET_COL].astype(int).to_numpy()
    y_valid = valid[TARGET_COL].astype(int).to_numpy()
    return train, valid, X_train, X_valid, y_train, y_valid, numeric_features, categorical_features


def fit_catboost_with_config(X_train, y_train, numeric_features, categorical_features, params: dict):
    if not HAS_CATBOOST:
        raise RuntimeError("catboost is not installed")
    X = X_train.copy().replace({pd.NA: np.nan})
    for column in numeric_features:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_features:
        if column in X.columns:
            X[column] = X[column].astype(object)
            X.loc[pd.isna(X[column]), column] = MISSING_CATEGORY
            X[column] = X[column].astype(str)
    cat_idx = [X.columns.get_loc(c) for c in categorical_features if c in X.columns]
    model = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=RANDOM_STATE,
    )
    model.fit(X, y_train, cat_features=cat_idx)
    return model


def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    train, valid, X_train, X_valid, y_train, y_valid, numeric_features, categorical_features = prepare_data(df)
    rows = []

    for config_id, params in catboost_configs().items():
        if not HAS_CATBOOST:
            rows.append({"model": "catboost", "config_id": config_id, "status": "catboost not installed"})
            continue
        model = fit_catboost_with_config(X_train, y_train, numeric_features, categorical_features, params)
        prob = predict_catboost(model, X_valid, numeric_features, categorical_features)
        row = {
            "model": "catboost",
            "config_id": config_id,
            "is_primary_fixed_config": is_primary_fixed_config(config_id),
            "inner_train_waves": INNER_TRAIN_WAVES,
            "inner_validation_waves": INNER_VALIDATION_WAVES,
            "inner_train_n": int(len(train)),
            "inner_validation_n": int(len(valid)),
            "parameters": json.dumps(params, sort_keys=True),
        }
        row.update(evaluate_probabilities(y_valid, prob))
        rows.append(row)

    for config_id, params in xgboost_configs().items():
        if not HAS_XGBOOST:
            rows.append({"model": "xgboost", "config_id": config_id, "status": "xgboost not installed"})
            continue
        estimator = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model = fit_sklearn_pipeline(estimator, X_train, y_train, numeric_features, categorical_features)
        prob = model.predict_proba(X_valid)[:, 1]
        row = {
            "model": "xgboost",
            "config_id": config_id,
            "is_primary_fixed_config": is_primary_fixed_config(config_id),
            "inner_train_waves": INNER_TRAIN_WAVES,
            "inner_validation_waves": INNER_VALIDATION_WAVES,
            "inner_train_n": int(len(train)),
            "inner_validation_n": int(len(valid)),
            "parameters": json.dumps(params, sort_keys=True),
        }
        row.update(evaluate_probabilities(y_valid, prob))
        rows.append(row)

    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run train-period hyperparameter sensitivity checks.")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = args.project_dir.resolve()
    output_dir = project_dir / "sr_additional_analyses" / "train_period_sensitivity"
    table_dir = project_dir / "outputs_klips_sr" / "tables"
    ensure_dir(output_dir)
    ensure_dir(table_dir)
    setup_logging(output_dir / "logs" / "13_train_period_sensitivity.log")

    df = load_analysis_base(project_dir)
    result = run_sensitivity(df)
    result_path = output_dir / "train_period_hyperparameter_sensitivity.csv"
    table_path = table_dir / "TableS30_train_period_hyperparameter_sensitivity.csv"
    table_columns = [
        "model",
        "config_id",
        "is_primary_fixed_config",
        "inner_train_waves",
        "inner_validation_waves",
        "roc_auc",
        "pr_auc",
        "brier",
        "recall_at_20",
        "lift_at_20",
    ]
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    result.reindex(columns=table_columns).to_csv(table_path, index=False, encoding="utf-8-sig")

    manifest = {
        "inner_train_waves": INNER_TRAIN_WAVES,
        "inner_validation_waves": INNER_VALIDATION_WAVES,
        "outputs": [str(result_path), str(table_path)],
        "note": "The held-out test waves remain untouched; configurations are compared only inside the original training period.",
    }
    with open(output_dir / "train_period_sensitivity_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
