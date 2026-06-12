from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline_lib.features import select_model_columns as shared_select_model_columns

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_analysis_base(project_dir: Path) -> pd.DataFrame:
    input_path = project_dir / "outputs_klips_sr" / "processed" / "analysis_base_with_label.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}. "
            "Run 01_build_analysis_base.py or 11_run_full_pipeline.py first."
        )
    return pd.read_csv(input_path)


def select_model_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    return shared_select_model_columns(df)


def sanitize_model_input(X: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    X = X.copy().replace({pd.NA: np.nan})
    for column in numeric_features:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_features:
        if column in X.columns:
            X[column] = X[column].astype(object)
            X.loc[pd.isna(X[column]), column] = np.nan
    return X


def make_sklearn_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    n_obs = len(y_true)
    top_n = max(1, int(np.ceil(n_obs * k)))
    top_index = np.argsort(-y_prob)[:top_n]
    positives_total = y_true.sum()
    if positives_total == 0:
        return np.nan
    return float(y_true[top_index].sum() / positives_total)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    base_rate = y_true.mean()
    if base_rate == 0:
        return np.nan
    n_obs = len(y_true)
    top_n = max(1, int(np.ceil(n_obs * k)))
    top_index = np.argsort(-y_prob)[:top_n]
    precision_top = y_true[top_index].mean()
    return float(precision_top / base_rate)


def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "recall_at_10": recall_at_k(y_true, y_prob, 0.10),
        "lift_at_10": lift_at_k(y_true, y_prob, 0.10),
        "recall_at_20": recall_at_k(y_true, y_prob, 0.20),
        "lift_at_20": lift_at_k(y_true, y_prob, 0.20),
    }


def threshold_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "positive_prediction_rate": float(np.mean(y_pred)),
    }


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def choose_threshold_for_top_share(y_prob: np.ndarray, share: float) -> float:
    y_prob = np.asarray(y_prob, dtype=float)
    share = float(share)
    if not (0 < share <= 1):
        raise ValueError("share must be within (0, 1].")
    n_obs = len(y_prob)
    top_n = max(1, int(np.ceil(n_obs * share)))
    sorted_scores = np.sort(y_prob)[::-1]
    return float(sorted_scores[top_n - 1])


def split_chronological(df: pd.DataFrame, train_end: int = 20, valid_end: int = 23):
    train = df[df["wave"] <= train_end].copy()
    valid = df[(df["wave"] > train_end) & (df["wave"] <= valid_end)].copy()
    test = df[df["wave"] > valid_end].copy()
    return train, valid, test


def split_random_exact(df: pd.DataFrame, n_valid: int = 25961, n_test: int = 27063, seed: int = 42):
    if n_valid + n_test >= len(df):
        raise ValueError("n_valid + n_test must be smaller than total row count.")
    train_valid, test = train_test_split(df, test_size=n_test, random_state=seed, stratify=df["exit_label_t1"])
    train, valid = train_test_split(train_valid, test_size=n_valid, random_state=seed, stratify=train_valid["exit_label_t1"])
    return train.copy(), valid.copy(), test.copy()


def chronological_overlap_audit(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "pid") -> pd.DataFrame:
    if id_col not in train_df.columns or id_col not in valid_df.columns or id_col not in test_df.columns:
        return pd.DataFrame()
    train_ids = set(train_df[id_col].dropna().tolist())
    valid_ids = set(valid_df[id_col].dropna().tolist())
    test_ids = set(test_df[id_col].dropna().tolist())

    rows = []
    for split_name, split_df, prior_ids in [
        ("valid", valid_df, train_ids),
        ("test_vs_train", test_df, train_ids),
        ("test_vs_train_valid", test_df, train_ids | valid_ids),
    ]:
        split_ids = split_df[id_col].dropna()
        if split_ids.empty:
            continue
        row_overlap_mask = split_ids.isin(prior_ids)
        unique_split_ids = set(split_ids.tolist())
        unique_overlap_ids = unique_split_ids & prior_ids
        rows.append({
            "comparison": split_name,
            "row_n": int(len(split_df)),
            "unique_pid_n": int(len(unique_split_ids)),
            "overlap_row_n": int(row_overlap_mask.sum()),
            "overlap_row_rate": float(row_overlap_mask.mean()),
            "overlap_unique_pid_n": int(len(unique_overlap_ids)),
            "overlap_unique_pid_rate": float(len(unique_overlap_ids) / len(unique_split_ids)) if unique_split_ids else np.nan,
        })
    return pd.DataFrame(rows)


def split_person_disjoint_chronological(df: pd.DataFrame, train_end: int = 20, valid_end: int = 23, id_col: str = "pid"):
    train = df[df["wave"] <= train_end].copy()
    valid_candidate = df[(df["wave"] > train_end) & (df["wave"] <= valid_end)].copy()
    test_candidate = df[df["wave"] > valid_end].copy()

    if id_col not in df.columns:
        return train, valid_candidate, test_candidate

    train_ids = set(train[id_col].dropna().tolist())
    valid = valid_candidate[~valid_candidate[id_col].isin(train_ids)].copy()
    valid_ids = set(valid[id_col].dropna().tolist())
    test = test_candidate[~test_candidate[id_col].isin(train_ids | valid_ids)].copy()
    return train, valid, test


def subset_new_person_rows(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "pid") -> pd.DataFrame:
    if id_col not in train_df.columns or id_col not in valid_df.columns or id_col not in test_df.columns:
        return pd.DataFrame()
    seen_ids = set(train_df[id_col].dropna().tolist()) | set(valid_df[id_col].dropna().tolist())
    return test_df[~test_df[id_col].isin(seen_ids)].copy()


def fit_sklearn_pipeline(estimator, X_train, y_train, numeric_features, categorical_features):
    preprocessor = make_sklearn_preprocessor(numeric_features, categorical_features)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    clf.fit(X_train, y_train)
    return clf


def fit_catboost_pipeline(X_train, y_train, numeric_features, categorical_features):
    if not HAS_CATBOOST:
        raise RuntimeError("catboost is not installed")
    X_train = X_train.copy().replace({pd.NA: np.nan})
    for column in numeric_features:
        if column in X_train.columns:
            X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
    for column in categorical_features:
        if column in X_train.columns:
            X_train[column] = X_train[column].astype(object)
            X_train.loc[pd.isna(X_train[column]), column] = "__MISSING__"
            X_train[column] = X_train[column].astype(str)
    cat_feature_index = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]
    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05, loss_function="Logloss",
        eval_metric="AUC", verbose=False, random_seed=42,
    )
    model.fit(X_train, y_train, cat_features=cat_feature_index)
    return model


def predict_catboost(model, X, numeric_features, categorical_features):
    X = X.copy().replace({pd.NA: np.nan})
    for column in numeric_features:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_features:
        if column in X.columns:
            X[column] = X[column].astype(object)
            X.loc[pd.isna(X[column]), column] = "__MISSING__"
            X[column] = X[column].astype(str)
    return model.predict_proba(X)[:, 1]


def available_base_models():
    models = {
        "logistic": ("sklearn", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")),
        "random_forest": (
            "sklearn",
            RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=5,
                random_state=42, n_jobs=-1, class_weight="balanced_subsample",
            ),
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = (
            "sklearn",
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, objective="binary:logistic", eval_metric="auc",
                random_state=42, n_jobs=-1,
            ),
        )
    if HAS_CATBOOST:
        models["catboost"] = ("catboost", None)
    return models


def fit_base_and_hybrid(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, split_scheme: str):
    feature_cols, numeric_features, categorical_features = select_model_columns(train_df)
    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    y_train = train_df["exit_label_t1"].astype(int).to_numpy()
    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    y_valid = valid_df["exit_label_t1"].astype(int).to_numpy()
    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)
    y_test = test_df["exit_label_t1"].astype(int).to_numpy()

    models = available_base_models()
    metrics_rows = []
    valid_pred_df = valid_df[[c for c in ["pid", "wave", "exit_label_t1"] if c in valid_df.columns]].copy()
    test_pred_df = test_df[[c for c in ["pid", "wave", "exit_label_t1"] if c in test_df.columns]].copy()

    for model_name, (model_type, estimator) in models.items():
        if model_type == "sklearn":
            model = fit_sklearn_pipeline(estimator, X_train, y_train, numeric_features, categorical_features)
            valid_prob = model.predict_proba(X_valid)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
        else:
            model = fit_catboost_pipeline(X_train, y_train, numeric_features, categorical_features)
            valid_prob = predict_catboost(model, X_valid, numeric_features, categorical_features)
            test_prob = predict_catboost(model, X_test, numeric_features, categorical_features)
        valid_pred_df[f"proba_{model_name}"] = valid_prob
        test_pred_df[f"proba_{model_name}"] = test_prob
        for split_name, y_true, y_prob in [("valid", y_valid, valid_prob), ("test", y_test, test_prob)]:
            row = {"split_scheme": split_scheme, "model": model_name, "dataset": split_name}
            row.update(evaluate_binary_classifier(y_true, y_prob))
            metrics_rows.append(row)

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_train = pd.DataFrame(index=train_df.index)
    valid_meta = pd.DataFrame(index=valid_df.index)
    test_meta = pd.DataFrame(index=test_df.index)

    for model_name, (model_type, estimator) in models.items():
        meta_col = f"meta_{model_name}"
        oof_pred = np.zeros(len(train_df))
        valid_fold_preds = []
        test_fold_preds = []
        for train_index, fold_valid_index in stratified_kfold.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_index].copy()
            y_fold_train = y_train[train_index]
            X_fold_valid = X_train.iloc[fold_valid_index].copy()
            if model_type == "sklearn":
                fold_model = fit_sklearn_pipeline(estimator, X_fold_train, y_fold_train, numeric_features, categorical_features)
                oof_pred[fold_valid_index] = fold_model.predict_proba(X_fold_valid)[:, 1]
                valid_fold_preds.append(fold_model.predict_proba(X_valid)[:, 1])
                test_fold_preds.append(fold_model.predict_proba(X_test)[:, 1])
            else:
                fold_model = fit_catboost_pipeline(X_fold_train, y_fold_train, numeric_features, categorical_features)
                oof_pred[fold_valid_index] = predict_catboost(fold_model, X_fold_valid, numeric_features, categorical_features)
                valid_fold_preds.append(predict_catboost(fold_model, X_valid, numeric_features, categorical_features))
                test_fold_preds.append(predict_catboost(fold_model, X_test, numeric_features, categorical_features))
        oof_train[meta_col] = oof_pred
        valid_meta[meta_col] = np.mean(np.column_stack(valid_fold_preds), axis=1)
        test_meta[meta_col] = np.mean(np.column_stack(test_fold_preds), axis=1)

    meta_model = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")
    meta_model.fit(oof_train, y_train)
    hybrid_valid_prob = meta_model.predict_proba(valid_meta)[:, 1]
    hybrid_test_prob = meta_model.predict_proba(test_meta)[:, 1]
    valid_pred_df["proba_hybrid_stack"] = hybrid_valid_prob
    test_pred_df["proba_hybrid_stack"] = hybrid_test_prob
    for split_name, y_true, y_prob in [("valid", y_valid, hybrid_valid_prob), ("test", y_test, hybrid_test_prob)]:
        row = {"split_scheme": split_scheme, "model": "hybrid_stack", "dataset": split_name}
        row.update(evaluate_binary_classifier(y_true, y_prob))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, valid_pred_df, test_pred_df


def fit_calibrators(valid_prob: np.ndarray, y_valid: np.ndarray):
    valid_prob = np.asarray(valid_prob, dtype=float)
    y_valid = np.asarray(y_valid, dtype=int)
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(valid_prob.reshape(-1, 1), y_valid)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(valid_prob, y_valid)
    return platt, isotonic


def apply_calibrators(test_prob: np.ndarray, platt, isotonic):
    test_prob = np.asarray(test_prob, dtype=float)
    platt_prob = platt.predict_proba(test_prob.reshape(-1, 1))[:, 1]
    isotonic_prob = isotonic.predict(test_prob)
    return platt_prob, isotonic_prob


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted": prob_pred, "observed": prob_true})


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: Iterable,
    metric_fn,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(list(groups))
    unique_groups = pd.Index(groups).dropna().unique().tolist()
    if len(unique_groups) < 2:
        return np.nan, np.nan, np.nan

    group_to_index = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_index = np.concatenate([group_to_index[g] for g in sampled_groups])
        y_boot = y_true[sampled_index]
        p_boot = y_prob[sampled_index]
        if len(np.unique(y_boot)) < 2:
            continue
        scores.append(metric_fn(y_boot, p_boot))
    if not scores:
        return np.nan, np.nan, np.nan
    scores = np.asarray(scores, dtype=float)
    return float(np.mean(scores)), float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def evaluate_prediction_subset(pred_df: pd.DataFrame, prob_cols: List[str], y_col: str = "exit_label_t1", threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    if pred_df.empty:
        return pd.DataFrame()
    y_true = pred_df[y_col].to_numpy(dtype=int)
    for prob_col in prob_cols:
        row = {"model": prob_col.replace("proba_", ""), "n": int(len(pred_df))}
        row.update(evaluate_binary_classifier(y_true, pred_df[prob_col].to_numpy(dtype=float), threshold=threshold))
        rows.append(row)
    return pd.DataFrame(rows)

