
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pipeline_lib.modeling import (
    HAS_CATBOOST,
    HAS_XGBOOST,
    evaluate_binary_classifier,
    fit_catboost_pipeline,
    fit_sklearn_pipeline,
    predict_catboost,
    sanitize_model_input,
    select_model_columns,
    setup_logging,
    split_chronological as timewise_split,
)
from pipeline_lib.paths import resolve_project_dir

try:
    from xgboost import XGBClassifier
except ImportError:                                                 
    XGBClassifier = None


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"

                                                                                
                                                                   
TRAIN_END_WAVE = 20
VALID_END_WAVE = 23

TARGET_COL = "exit_label_t1"
ID_COLS = ("pid", "wave")
RANDOM_STATE = 42

setup_logging(LOG_DIR / "klips_stage2_multimodel.log")
logger = logging.getLogger(__name__)


def _evaluate_splits(model_name: str, splits) -> pd.DataFrame:
    rows = []
    for split_name, y_true, y_prob in splits:
        row = {"model": model_name, "split": split_name}
        row.update(evaluate_binary_classifier(y_true, y_prob))
        rows.append(row)
    return pd.DataFrame(rows)


def _test_prediction_frame(test_df: pd.DataFrame, model_name: str, test_prob, target_col: str = TARGET_COL) -> pd.DataFrame:
    keep = [column for column in (*ID_COLS, target_col) if column in test_df.columns]
    frame = test_df[keep].copy() if keep else pd.DataFrame(index=test_df.index)
    frame[f"proba_{model_name}"] = test_prob
    return frame


def fit_sklearn_model(
    model_name: str,
    estimator,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)

    y_train = train_df[target_col].astype(int)
    y_valid = valid_df[target_col].astype(int).to_numpy()
    y_test = test_df[target_col].astype(int).to_numpy()

    clf = fit_sklearn_pipeline(estimator, X_train, y_train, numeric_features, categorical_features)
    valid_prob = clf.predict_proba(X_valid)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    metrics = _evaluate_splits(model_name, [("valid", y_valid, valid_prob), ("test", y_test, test_prob)])
    predictions = _test_prediction_frame(test_df, model_name, test_prob, target_col)
    return metrics, predictions


def fit_catboost_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = train_df[target_col].astype(int)
    y_valid = valid_df[target_col].astype(int).to_numpy()
    y_test = test_df[target_col].astype(int).to_numpy()

    model = fit_catboost_pipeline(train_df[feature_cols], y_train, numeric_features, categorical_features)
    valid_prob = predict_catboost(model, valid_df[feature_cols], numeric_features, categorical_features)
    test_prob = predict_catboost(model, test_df[feature_cols], numeric_features, categorical_features)

    metrics = _evaluate_splits("catboost", [("valid", y_valid, valid_prob), ("test", y_test, test_prob)])
    predictions = _test_prediction_frame(test_df, "catboost", test_prob, target_col)
    return metrics, predictions


def build_sklearn_estimators() -> list[tuple[str, object]]:
    estimators: list[tuple[str, object]] = [
        ("logistic", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        ),
    ]

    if HAS_XGBOOST:
        estimators.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
    else:
        logger.warning("xgboost is not installed. XGBoost training is skipped.")

    return estimators


def main() -> None:
    logger.info("==== Stage 2 multimodel training start ====")

    input_path = PROCESSED_DIR / "analysis_base_with_label.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Loaded analysis base: %s", df.shape)

    train_df, valid_df, test_df = timewise_split(df, train_end=TRAIN_END_WAVE, valid_end=VALID_END_WAVE)
    logger.info("Train=%s, Valid=%s, Test=%s", train_df.shape, valid_df.shape, test_df.shape)

    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    logger.info("Feature count=%s", len(feature_cols))
    logger.info("Numeric features=%s", numeric_features)
    logger.info("Categorical features=%s", categorical_features)

    metrics_all: list[pd.DataFrame] = []
    predictions_all: list[pd.DataFrame] = []

    for model_name, estimator in build_sklearn_estimators():
        metrics, predictions = fit_sklearn_model(
            model_name=model_name,
            estimator=estimator,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        metrics_all.append(metrics)
        predictions_all.append(predictions)

    if HAS_CATBOOST:
        cb_metrics, cb_predictions = fit_catboost_model(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        metrics_all.append(cb_metrics)
        predictions_all.append(cb_predictions)
    else:
        logger.warning("catboost is not installed. CatBoost training is skipped.")

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "stage2_model_metrics.csv", index=False, encoding="utf-8-sig")

                                                                              
    prediction_base = predictions_all[0].copy()
    for additional_df in predictions_all[1:]:
        merge_cols = [c for c in (*ID_COLS, TARGET_COL) if c in prediction_base.columns and c in additional_df.columns]
        score_cols = [c for c in additional_df.columns if c.startswith("proba_")]
        prediction_base = prediction_base.merge(additional_df[merge_cols + score_cols], on=merge_cols, how="left")
    prediction_base.to_csv(OUTPUT_DIR / "stage2_test_predictions_all_models.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_path": str(input_path),
        "train_shape": list(train_df.shape),
        "valid_shape": list(valid_df.shape),
        "test_shape": list(test_df.shape),
        "feature_count": len(feature_cols),
        "models_run": metrics_df["model"].drop_duplicates().tolist(),
    }
    with open(OUTPUT_DIR / "stage2_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logger.info("==== Stage 2 multimodel training end ====")
    logger.info("\n%s", metrics_df)


if __name__ == "__main__":
    main()
