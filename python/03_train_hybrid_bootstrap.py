
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

N_FOLDS = 5                                     
N_BOOTSTRAP = 300                                                  
N_CALIBRATION_BINS = 10                                           
BOOTSTRAP_SEED = 42

                                                                                          
CI_METRICS: tuple[tuple[str, object], ...] = (
    ("roc_auc", roc_auc_score),
    ("pr_auc", average_precision_score),
    ("brier", brier_score_loss),
)

setup_logging(LOG_DIR / "klips_stage3_hybrid.log")
logger = logging.getLogger(__name__)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    scores = []

    for _ in range(n_boot):
        sample_index = rng.integers(0, n_obs, n_obs)
        y_boot = y_true[sample_index]
        if np.unique(y_boot).size < 2:
            continue
        scores.append(metric_fn(y_boot, y_prob[sample_index]))

    if not scores:
        return np.nan, np.nan, np.nan

    scores = np.asarray(scores)
    return float(scores.mean()), float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def save_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, out_dir: Path, n_bins: int = N_CALIBRATION_BINS) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).to_csv(
        out_dir / f"calibration_{model_name}.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve - {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"calibration_{model_name}.png", dpi=200)
    plt.close()


def build_base_models() -> dict[str, tuple[str, object]]:
    models: dict[str, tuple[str, object]] = {
        "logistic": ("sklearn", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", tol=1e-4)),
        "random_forest": (
            "sklearn",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = (
            "sklearn",
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
    if HAS_CATBOOST:
        models["catboost"] = ("catboost", None)
    return models


def main() -> None:
    logger.info("==== Stage 3 hybrid stacking start ====")

    input_path = PROCESSED_DIR / "analysis_base_with_label.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Loaded analysis base: %s", df.shape)

    train_df, valid_df, test_df = timewise_split(df, train_end=TRAIN_END_WAVE, valid_end=VALID_END_WAVE)
    logger.info("Train=%s, Valid=%s, Test=%s", train_df.shape, valid_df.shape, test_df.shape)

    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    logger.info("Feature count=%s", len(feature_cols))

    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)

    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_valid = valid_df[TARGET_COL].astype(int).to_numpy()
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    base_models = build_base_models()
    stratified_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

                                                                                    
                                                               
    oof_train = pd.DataFrame(index=train_df.index)
    valid_meta = pd.DataFrame(index=valid_df.index)
    test_meta = pd.DataFrame(index=test_df.index)
    metrics_rows = []

    for model_name, (backend, estimator) in base_models.items():
        logger.info("Training base model: %s", model_name)

        meta_col = f"meta_{model_name}"
        oof_pred = np.zeros(len(train_df))
        valid_fold_preds = []
        test_fold_preds = []

        for train_index, fold_valid_index in stratified_kfold.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_index].copy()
            y_fold_train = y_train[train_index]
            X_fold_valid = X_train.iloc[fold_valid_index].copy()

            if backend == "sklearn":
                model = fit_sklearn_pipeline(estimator, X_fold_train, y_fold_train, numeric_features, categorical_features)
                oof_pred[fold_valid_index] = model.predict_proba(X_fold_valid)[:, 1]
                valid_fold_preds.append(model.predict_proba(X_valid)[:, 1])
                test_fold_preds.append(model.predict_proba(X_test)[:, 1])
            elif backend == "catboost":
                model = fit_catboost_pipeline(X_fold_train, y_fold_train, numeric_features, categorical_features)
                oof_pred[fold_valid_index] = predict_catboost(model, X_fold_valid, numeric_features, categorical_features)
                valid_fold_preds.append(predict_catboost(model, X_valid, numeric_features, categorical_features))
                test_fold_preds.append(predict_catboost(model, X_test, numeric_features, categorical_features))

                                                                          
        valid_pred = np.mean(np.column_stack(valid_fold_preds), axis=1)
        test_pred = np.mean(np.column_stack(test_fold_preds), axis=1)

        oof_train[meta_col] = oof_pred
        valid_meta[meta_col] = valid_pred
        test_meta[meta_col] = test_pred

        for split_name, y_true, y_prob in [("valid", y_valid, valid_pred), ("test", y_test, test_pred)]:
            row = {"model": model_name, "split": split_name}
            row.update(evaluate_binary_classifier(y_true, y_prob))
            metrics_rows.append(row)

                                                          
    meta_feature_order = oof_train.columns.tolist()
    meta_features_train = oof_train[meta_feature_order]
    meta_features_valid = valid_meta[meta_feature_order]
    meta_features_test = test_meta[meta_feature_order]

    meta_model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", tol=1e-4)
    meta_model.fit(meta_features_train, y_train)

    hybrid_valid_prob = meta_model.predict_proba(meta_features_valid)[:, 1]
    hybrid_test_prob = meta_model.predict_proba(meta_features_test)[:, 1]

    for split_name, y_true, y_prob in [("valid", y_valid, hybrid_valid_prob), ("test", y_test, hybrid_test_prob)]:
        row = {"model": "hybrid_stack", "split": split_name}
        row.update(evaluate_binary_classifier(y_true, y_prob))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "stage3_hybrid_metrics.csv", index=False, encoding="utf-8-sig")

                                                                             
    test_prob_by_model = {col.replace("meta_", ""): test_meta[col].to_numpy() for col in test_meta.columns}
    test_prob_by_model["hybrid_stack"] = hybrid_test_prob

    ci_rows = []
    for model_name, prob in test_prob_by_model.items():
        for metric_name, metric_fn in CI_METRICS:
            mean_val, ci_low, ci_high = bootstrap_ci(y_test, prob, metric_fn, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            ci_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "bootstrap_mean": mean_val,
                    "ci_2.5": ci_low,
                    "ci_97.5": ci_high,
                }
            )
    pd.DataFrame(ci_rows).to_csv(OUTPUT_DIR / "stage3_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    for model_name, prob in test_prob_by_model.items():
        save_calibration_curve(y_test, prob, model_name, OUTPUT_DIR, n_bins=N_CALIBRATION_BINS)

                                                                                
    valid_pred_out = valid_df[[c for c in (*ID_COLS, TARGET_COL) if c in valid_df.columns]].copy()
    valid_prob_by_model = {col.replace("meta_", ""): valid_meta[col].to_numpy() for col in valid_meta.columns}
    valid_prob_by_model["hybrid_stack"] = hybrid_valid_prob
    for model_name, prob in valid_prob_by_model.items():
        valid_pred_out[f"proba_{model_name}"] = prob
    valid_pred_out.to_csv(OUTPUT_DIR / "stage3_valid_predictions_with_hybrid.csv", index=False, encoding="utf-8-sig")

                                                                                  
    pred_out = test_df[[c for c in (*ID_COLS, TARGET_COL) if c in test_df.columns]].copy()
    for model_name, prob in test_prob_by_model.items():
        pred_out[f"proba_{model_name}"] = prob
    pred_out.to_csv(OUTPUT_DIR / "stage3_test_predictions_with_hybrid.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame({"meta_feature": meta_feature_order, "coefficient": meta_model.coef_[0]})
    coef_df.to_csv(OUTPUT_DIR / "stage3_hybrid_meta_coefficients.csv", index=False, encoding="utf-8-sig")

    summary = {
        "train_shape": list(train_df.shape),
        "valid_shape": list(valid_df.shape),
        "test_shape": list(test_df.shape),
        "base_models": list(base_models.keys()),
        "hybrid_features": meta_feature_order,
    }
    with open(OUTPUT_DIR / "stage3_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logger.info("==== Stage 3 hybrid stacking end ====")
    logger.info("\n%s", metrics_df)


if __name__ == "__main__":
    main()
