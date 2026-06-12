
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier

    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

from pipeline_lib.features import select_model_columns
from pipeline_lib.paths import resolve_project_dir


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
PROCESSED_DIR = OUTPUT_DIR / "processed"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = OUTPUT_DIR / "models"
ADDITIONAL_DIR = PROJECT_DIR / "sr_additional_analyses"

for directory in [FIG_DIR, TAB_DIR, LOG_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


MODEL_ORDER = ["logistic", "random_forest", "xgboost", "catboost", "hybrid_stack"]
MODEL_LABELS = {
    "logistic": "Logistic",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "hybrid_stack": "Hybrid stack",
}

                                                                              
RANDOM_STATE = 42
SHAP_SAMPLE_N = 5000                                                           
MISSING_CATEGORY = "__MISSING__"                                                

                                                   
TRAIN_END_WAVE = 20
VALID_WAVES = (21, 23)
TEST_WAVES = (24, 26)


def read_analysis_base() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "analysis_base_with_label.csv")


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["wave"] <= TRAIN_END_WAVE].copy()
    valid = df[(df["wave"] >= VALID_WAVES[0]) & (df["wave"] <= VALID_WAVES[1])].copy()
    test = df[(df["wave"] >= TEST_WAVES[0]) & (df["wave"] <= TEST_WAVES[1])].copy()
    return train, valid, test


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_pred = pd.read_csv(OUTPUT_DIR / "stage3_test_predictions_with_hybrid.csv")
    valid_pred_path = ADDITIONAL_DIR / "ablation_random_vs_chronological" / "chronological_valid_predictions.csv"
    valid_pred = pd.read_csv(valid_pred_path)
    return valid_pred, test_pred


def make_eda(df: pd.DataFrame) -> None:
    required = ["exit_label_t1", "age_final", "monthly_wage", "tenure_years", "job_satisfaction_mean", "weekly_hours"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EDA columns: {missing}")

    summary_rows = []
    for col in required[1:]:
        for y, g in df.groupby("exit_label_t1"):
            x = pd.to_numeric(g[col], errors="coerce").dropna()
            summary_rows.append(
                {
                    "feature": col,
                    "exit_label_t1": int(y),
                    "n": int(len(x)),
                    "mean": float(x.mean()) if len(x) else np.nan,
                    "sd": float(x.std()) if len(x) else np.nan,
                    "median": float(x.median()) if len(x) else np.nan,
                    "p25": float(x.quantile(0.25)) if len(x) else np.nan,
                    "p75": float(x.quantile(0.75)) if len(x) else np.nan,
                }
            )
    pd.DataFrame(summary_rows).to_csv(TAB_DIR / "TableS_EDA_summary_by_exit.csv", index=False, encoding="utf-8-sig")

    features = required[1:]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=300)
    axes = axes.ravel()
    for ax, col in zip(axes, features):
        for y, color in [(0, "#4C78A8"), (1, "#F58518")]:
            vals = pd.to_numeric(df.loc[df["exit_label_t1"] == y, col], errors="coerce").dropna()
            if vals.empty:
                continue
            lo, hi = vals.quantile(0.01), vals.quantile(0.99)
            vals = vals.clip(lo, hi)
            ax.hist(vals, bins=40, alpha=0.48, density=True, label=f"exit={y}", color=color)
        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle("Distributions of selected numeric predictors by one-year wage-employment exit", y=0.995, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS6_eda_by_exit.png", bbox_inches="tight")
    plt.close(fig)

    for col in features:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        for y, color in [(0, "#4C78A8"), (1, "#F58518")]:
            vals = pd.to_numeric(df.loc[df["exit_label_t1"] == y, col], errors="coerce").dropna()
            if vals.empty:
                continue
            vals = vals.clip(vals.quantile(0.01), vals.quantile(0.99))
            ax.hist(vals, bins=40, alpha=0.48, density=True, label=f"exit={y}", color=color)
        ax.set_title(f"Distribution of {col} by wage-employment exit")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"FigS6_EDA_{col}.png", bbox_inches="tight")
        plt.close(fig)


def make_roc_pr_lift_confusion(valid_pred: pd.DataFrame, test_pred: pd.DataFrame) -> None:
    y = test_pred["exit_label_t1"].astype(int).to_numpy()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    for model, color in zip(MODEL_ORDER, colors):
        prob = test_pred[f"proba_{model}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[model]} ({auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves in the held-out chronological test partition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig2_roc_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    base = y.mean()
    for model, color in zip(MODEL_ORDER, colors):
        prob = test_pred[f"proba_{model}"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(recall, precision, label=f"{MODEL_LABELS[model]} ({ap:.3f})", color=color)
    ax.axhline(base, linestyle="--", color="grey", linewidth=1, label=f"Baseline ({base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves in the held-out chronological test partition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig3_pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    decile_rows = []
    for model in MODEL_ORDER:
        tmp = pd.DataFrame({"y": y, "score": test_pred[f"proba_{model}"].to_numpy()}).sort_values("score", ascending=False)
        tmp["decile"] = pd.qcut(np.arange(len(tmp)), 10, labels=False) + 1
        for d, g in tmp.groupby("decile"):
            event_rate = g["y"].mean()
            decile_rows.append(
                {
                    "model": model,
                    "decile": int(d),
                    "event_rate": float(event_rate),
                    "lift": float(event_rate / base) if base else np.nan,
                }
            )
    dec = pd.DataFrame(decile_rows)
    dec.to_csv(TAB_DIR / "TableS_lift_decile.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for model, color in zip(MODEL_ORDER, colors):
        g = dec[dec["model"] == model]
        ax.plot(g["decile"], g["lift"], marker="o", label=MODEL_LABELS[model], color=color)
    ax.set_xlabel("Risk-score decile, 1 = highest risk")
    ax.set_ylabel("Lift over baseline event rate")
    ax.set_title("Decile lift in the held-out chronological test partition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS7_lift_decile_chart.png", bbox_inches="tight")
    plt.close(fig)

    threshold_path = ADDITIONAL_DIR / "posthoc_recalibration" / "recalibration_threshold_diagnostics.csv"
    threshold_df = pd.read_csv(threshold_path)
    rows = []
    for model in MODEL_ORDER:
        raw_rows = threshold_df[(threshold_df["model"] == model) & (threshold_df["calibration_method"] == "raw")]
        selected = raw_rows[raw_rows["threshold_rule"].isin(["default_0.5", "best_f1_valid", "top10_valid_cutoff", "top20_valid_cutoff"])]
        for _, row in selected.iterrows():
            th = float(row["selected_threshold"])
            pred_y = (test_pred[f"proba_{model}"].to_numpy() >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred_y, labels=[0, 1]).ravel()
            rows.append(
                {
                    "model": model,
                    "threshold_rule": row["threshold_rule"],
                    "selected_threshold": th,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                    "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                    "positive_prediction_rate": float((tp + fp) / len(y)),
                }
            )
    cm_df = pd.DataFrame(rows)
    cm_df.to_csv(TAB_DIR / "TableS_threshold_confusion_matrix.csv", index=False, encoding="utf-8-sig")

    cat_best = cm_df[(cm_df["model"] == "catboost") & (cm_df["threshold_rule"] == "best_f1_valid")].iloc[0]
    cm = np.array([[cat_best["tn"], cat_best["fp"]], [cat_best["fn"], cat_best["tp"]]], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    ax = axes[0, 0]
    for model, color in zip(MODEL_ORDER, colors):
        prob = test_pred[f"proba_{model}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=MODEL_LABELS[model], color=color)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("ROC")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    ax = axes[0, 1]
    for model, color in zip(MODEL_ORDER, colors):
        prob = test_pred[f"proba_{model}"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, prob)
        ax.plot(recall, precision, label=MODEL_LABELS[model], color=color)
    ax.axhline(base, linestyle="--", color="grey", lw=1)
    ax.set_title("Precision-recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax = axes[1, 0]
    for model, color in zip(MODEL_ORDER, colors):
        g = dec[dec["model"] == model]
        ax.plot(g["decile"], g["lift"], marker="o", label=MODEL_LABELS[model], color=color)
    ax.set_title("Decile lift")
    ax.set_xlabel("Risk decile")
    ax.set_ylabel("Lift")

    ax = axes[1, 1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Observed 0", "Observed 1"])
    ax.set_title("CatBoost confusion matrix\nvalidation-selected F1 threshold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / "Fig2_roc_pr_lift_confusion.png", bbox_inches="tight")
    plt.close(fig)


def prepare_catboost_matrix(df: pd.DataFrame, feature_cols: list[str], numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy().replace({pd.NA: np.nan})
    for col in numeric_features:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical_features:
        if col in out.columns:
            out[col] = out[col].astype(object)
            out.loc[pd.isna(out[col]), col] = MISSING_CATEGORY
            out[col] = out[col].astype(str)
    return out


def short_feature_value(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, str):
        return value[:35]
    try:
        v = float(value)
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    except (TypeError, ValueError):
        return str(value)[:35]


def make_shap_local_and_dependence(df: pd.DataFrame) -> None:
    train, _, test = chronological_split(df)
    feature_cols, numeric_features, categorical_features = select_model_columns(train)
    X_train = prepare_catboost_matrix(train, feature_cols, numeric_features, categorical_features)
    y_train = train["exit_label_t1"].astype(int)
    X_test = prepare_catboost_matrix(test, feature_cols, numeric_features, categorical_features)

    cat_idx = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=RANDOM_STATE,
        thread_count=-1,
    )
    model.fit(X_train, y_train, cat_features=cat_idx)
    model.save_model(str(MODEL_DIR / "catboost_model_additional_analyses.cbm"))

    sample_n = min(SHAP_SAMPLE_N, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
    sample_pool = Pool(X_sample, cat_features=cat_idx)
    shap_matrix = model.get_feature_importance(sample_pool, type="ShapValues")
    shap_values = shap_matrix[:, :-1]
    base_values = shap_matrix[:, -1]
    score = pd.Series(model.predict_proba(X_sample)[:, 1], index=X_sample.index)
    feature_names = list(X_sample.columns)
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index)

    case_index = {
        "high_risk": score.sort_values(ascending=False).index[0],
        "mid_risk": (score - score.median()).abs().sort_values().index[0],
        "low_risk": score.sort_values(ascending=True).index[0],
    }
    case_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
    for ax, (case_type, idx) in zip(axes, case_index.items()):
        contrib = shap_df.loc[idx].sort_values(key=lambda s: s.abs(), ascending=False).head(12).sort_values()
        colors = np.where(contrib.values >= 0, "#E45756", "#4C78A8")
        ax.barh(range(len(contrib)), contrib.values, color=colors)
        ax.set_yticks(range(len(contrib)), labels=contrib.index, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{case_type.replace('_', ' ').title()}\np={score.loc[idx]:.3f}", fontsize=9)
        ax.set_xlabel("SHAP contribution")
        positives = shap_df.loc[idx].sort_values(ascending=False).head(5)
        negatives = shap_df.loc[idx].sort_values(ascending=True).head(5)
        case_rows.append(
            {
                "case_type": case_type,
                "anonymous_case_id": f"case_{len(case_rows) + 1}",
                "predicted_probability": float(score.loc[idx]),
                "risk_score_quantile_in_sample": float(score.rank(pct=True).loc[idx]),
                "top_positive_features": "; ".join([f"{k}={short_feature_value(X_sample.loc[idx, k])}" for k in positives.index]),
                "top_negative_features": "; ".join([f"{k}={short_feature_value(X_sample.loc[idx, k])}" for k in negatives.index]),
            }
        )
    fig.suptitle("Local CatBoost SHAP explanations for high-, mid-, and low-risk anonymous cases", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig4_shap_waterfall_high_mid_low.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(case_rows).to_csv(TAB_DIR / "TableS_SHAP_local_cases.csv", index=False, encoding="utf-8-sig")

    for feature, filename, title in [
        ("age_squared", "Fig5_shap_dependence_age_squared.png", "SHAP dependence: age_squared"),
        ("job_sat_stability", "FigS8_shap_dependence_job_stability.png", "SHAP dependence: job_sat_stability"),
        ("wage_to_industry_wave_median", "FigS8_shap_dependence_wage_to_industry_wave_median.png", "SHAP dependence: wage-to-industry median ratio"),
    ]:
        if feature not in X_sample.columns:
            continue
        x = pd.to_numeric(X_sample[feature], errors="coerce")
        y = shap_df[feature]
        valid = x.notna() & y.notna()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.scatter(x[valid], y[valid], s=8, alpha=0.35, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        plt.close(fig)

    try:
        interactions = model.get_feature_importance(Pool(X_sample, cat_features=cat_idx), type="Interaction")
        interaction_rows = []
        for row in interactions[:20]:
            f1, f2, strength = int(row[0]), int(row[1]), float(row[2])
            interaction_rows.append({"feature_1": feature_names[f1], "feature_2": feature_names[f2], "interaction_strength": strength})
        interaction_df = pd.DataFrame(interaction_rows)
    except (RuntimeError, ValueError, TypeError):
        top_features = shap_df.abs().mean().sort_values(ascending=False).head(12).index.tolist()
        rows = []
        encoded = X_sample[top_features].copy()
        for col in encoded.columns:
            encoded[col] = pd.factorize(encoded[col])[0] if encoded[col].dtype == object else pd.to_numeric(encoded[col], errors="coerce")
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i + 1 :]:
                rows.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "interaction_strength": float(abs(encoded[f1].corr(encoded[f2])) * (shap_df[f1].abs().mean() + shap_df[f2].abs().mean())),
                    }
                )
        interaction_df = pd.DataFrame(rows).sort_values("interaction_strength", ascending=False).head(20)
    interaction_df.to_csv(TAB_DIR / "TableS_SHAP_interaction_top_pairs.csv", index=False, encoding="utf-8-sig")

    top_inter = interaction_df.head(10).copy()
    top_inter["pair"] = top_inter["feature_1"] + " x " + top_inter["feature_2"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.barh(top_inter["pair"][::-1], top_inter["interaction_strength"][::-1], color="#72B7B2")
    ax.set_xlabel("Interaction strength")
    ax.set_title("Top CatBoost interaction pairs in the SHAP sample")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS9_shap_interaction_top_pairs.png", bbox_inches="tight")
    plt.close(fig)


def make_ordinal_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def evaluate_probability(y_true, prob) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def choose_best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if thresholds.size == 0:
        return 0.5
    denom = precision[:-1] + recall[:-1]
    scores = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(scores))])


def recall_lift_at_share(y_true: np.ndarray, prob: np.ndarray, share: float) -> tuple[float, float]:
    n_obs = len(y_true)
    top_n = max(1, int(np.ceil(n_obs * share)))
    idx = np.argsort(-prob)[:top_n]
    positives = float(np.sum(y_true))
    if positives <= 0:
        return np.nan, np.nan
    recall = float(np.sum(y_true[idx]) / positives)
    base_rate = float(np.mean(y_true))
    lift = float(np.mean(y_true[idx]) / base_rate) if base_rate > 0 else np.nan
    return recall, lift


def make_tabnet_optional_baseline(df: pd.DataFrame) -> None:
    output_path = TAB_DIR / "TableS32_optional_deep_tabular_baseline.csv"
    columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "brier",
        "recall_at_20",
        "lift_at_20",
        "f1_at_valid_threshold",
        "valid_selected_threshold",
        "train_seconds",
        "test_inference_seconds",
        "note",
    ]
    if not HAS_TABNET:
        pd.DataFrame(
            [
                {
                    "model": "TabNet_optional",
                    "roc_auc": "",
                    "pr_auc": "",
                    "brier": "",
                    "recall_at_20": "",
                    "lift_at_20": "",
                    "f1_at_valid_threshold": "",
                    "valid_selected_threshold": "",
                    "train_seconds": "",
                    "test_inference_seconds": "",
                    "note": "pytorch-tabnet is not installed in the current Python environment.",
                }
            ],
            columns=columns,
        ).to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    train, valid, test = chronological_split(df)
    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    frames = []
    for frame in [train[feature_cols], valid[feature_cols], test[feature_cols]]:
        x = frame.copy()
        for column in numeric_features:
            x[column] = pd.to_numeric(x[column], errors="coerce")
        for column in categorical_features:
            x[column] = x[column].astype(object)
            x.loc[pd.isna(x[column]), column] = MISSING_CATEGORY
            x[column] = x[column].astype(str)
        frames.append(x)
    preprocessor = make_ordinal_preprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(frames[0]).astype(np.float32)
    X_valid = preprocessor.transform(frames[1]).astype(np.float32)
    X_test = preprocessor.transform(frames[2]).astype(np.float32)
    y_train = train["exit_label_t1"].astype(int).to_numpy()
    y_valid = valid["exit_label_t1"].astype(int).to_numpy()
    y_test = test["exit_label_t1"].astype(int).to_numpy()
    model = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_params={"lr": 0.02},
        seed=RANDOM_STATE,
        verbose=0,
        device_name="auto",
    )
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["auc"],
        max_epochs=80,
        patience=10,
        batch_size=2048,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
    )
    train_seconds = time.perf_counter() - start
    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold = choose_best_f1_threshold(y_valid, valid_prob)
    start = time.perf_counter()
    test_prob = model.predict_proba(X_test)[:, 1]
    inference_seconds = time.perf_counter() - start
    recall20, lift20 = recall_lift_at_share(y_test, test_prob, 0.20)
    row = {
        "model": "TabNet_optional",
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "brier": float(brier_score_loss(y_test, test_prob)),
        "recall_at_20": recall20,
        "lift_at_20": lift20,
        "f1_at_valid_threshold": float(f1_score(y_test, (test_prob >= threshold).astype(int), zero_division=0)),
        "valid_selected_threshold": threshold,
        "train_seconds": float(train_seconds),
        "test_inference_seconds": float(inference_seconds),
        "note": "Optional TabNet baseline trained under the same chronological split.",
    }
    pd.DataFrame([row], columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")


def make_oversampling_ablation(df: pd.DataFrame) -> None:
    if not HAS_IMBLEARN:
        pd.DataFrame([{"status": "imblearn not installed"}]).to_csv(TAB_DIR / "TableS18_oversampling_ablation.csv", index=False)
        return

    train, _, test = chronological_split(df)
    feature_cols, numeric_features, categorical_features = select_model_columns(train)
    preprocessor = make_ordinal_preprocessor(numeric_features, categorical_features)
    train_x = train[feature_cols].copy()
    test_x = test[feature_cols].copy()
    for col in numeric_features:
        if col in train_x.columns:
            train_x[col] = pd.to_numeric(train_x[col], errors="coerce")
            test_x[col] = pd.to_numeric(test_x[col], errors="coerce")
    for col in categorical_features:
        if col in train_x.columns:
            train_x[col] = train_x[col].astype(object)
            test_x[col] = test_x[col].astype(object)
            train_x.loc[pd.isna(train_x[col]), col] = MISSING_CATEGORY
            test_x.loc[pd.isna(test_x[col]), col] = MISSING_CATEGORY
            train_x[col] = train_x[col].astype(str)
            test_x[col] = test_x[col].astype(str)
    X_train = preprocessor.fit_transform(train_x)
    y_train = train["exit_label_t1"].astype(int).to_numpy()
    X_test = preprocessor.transform(test_x)
    y_test = test["exit_label_t1"].astype(int).to_numpy()

    strategies = {
        "none": None,
        "random_over": RandomOverSampler(random_state=RANDOM_STATE),
        "random_under": RandomUnderSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
        "adasyn": ADASYN(random_state=RANDOM_STATE, n_neighbors=5),
    }
    rows = []
    for name, sampler in strategies.items():
        try:
            if sampler is None:
                X_res, y_res = X_train, y_train
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            clf.fit(X_res, y_res)
            prob = clf.predict_proba(X_test)[:, 1]
            row = {
                "model": "xgboost_ordinal_encoded",
                "imbalance_strategy": name,
                "train_n_after_sampling": int(len(y_res)),
                "positive_rate_after_sampling": float(np.mean(y_res)),
                "note": "Sampling applied to training partition only; validation/test were not resampled.",
            }
            row.update(evaluate_probability(y_test, prob))
            rows.append(row)
        except (RuntimeError, ValueError, TypeError) as exc:
            rows.append(
                {
                    "model": "xgboost_ordinal_encoded",
                    "imbalance_strategy": name,
                    "status": f"failed: {type(exc).__name__}: {str(exc)[:180]}",
                    "note": "Sampling applied to training partition only; validation/test were not resampled.",
                }
            )
    pd.DataFrame(rows).to_csv(TAB_DIR / "TableS18_oversampling_ablation.csv", index=False, encoding="utf-8-sig")


def make_survival_baseline(df: pd.DataFrame) -> None:
    features = [
        "age_final",
        "monthly_wage",
        "tenure_years",
        "weekly_hours",
        "job_satisfaction_mean",
        "one_person_firm_flag",
        "gender",
        "education_level",
    ]
    available_features = [c for c in features if c in df.columns]
    rows = []
    for pid, g in df.sort_values(["pid", "wave"]).groupby("pid"):
        first = g.iloc[0]
        events = g[g["exit_label_t1"] == 1]
        if events.empty:
            event = 0
            duration = int(g["wave"].max() - first["wave"] + 1)
        else:
            event = 1
            duration = int(events["wave"].iloc[0] - first["wave"] + 1)
        row = {"pid": pid, "first_wave": int(first["wave"]), "duration_waves": max(duration, 1), "event": event}
        for col in available_features:
            row[col] = first[col]
        rows.append(row)
    base = pd.DataFrame(rows)
    base.to_csv(TAB_DIR / "person_level_survival_base.csv", index=False, encoding="utf-8-sig")

    cox_df = base.drop(columns=["pid"]).copy()
    for col in ["gender", "education_level"]:
        if col in cox_df.columns:
            cox_df[col] = cox_df[col].astype("category")
    cox_df = pd.get_dummies(cox_df, columns=[c for c in ["gender", "education_level"] if c in cox_df.columns], drop_first=True)
    for col in cox_df.columns:
        cox_df[col] = pd.to_numeric(cox_df[col], errors="coerce")
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()

    train = cox_df[cox_df["first_wave"] <= 20].drop(columns=["first_wave"])
    holdout = cox_df[cox_df["first_wave"] >= 21].drop(columns=["first_wave"])
    result_rows = []
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train, duration_col="duration_waves", event_col="event")
        train_risk = cph.predict_partial_hazard(train).to_numpy().ravel()
        holdout_risk = cph.predict_partial_hazard(holdout).to_numpy().ravel()
        train_c = concordance_index(train["duration_waves"], -train_risk, train["event"])
        holdout_c = concordance_index(holdout["duration_waves"], -holdout_risk, holdout["event"])
        result_rows.append(
            {
                "model": "cox_ph",
                "metric": "c_index",
                "train_value": float(train_c),
                "holdout_value": float(holdout_c),
                "train_n": int(len(train)),
                "holdout_n": int(len(holdout)),
                "note": "Fitted on persons whose first risk-set wave was <=20 and evaluated on later first-risk-wave persons.",
            }
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        result_rows.append({"model": "cox_ph", "metric": "status", "holdout_value": np.nan, "note": f"failed: {type(exc).__name__}: {str(exc)[:180]}"})
    res = pd.DataFrame(result_rows)
    res.to_csv(TAB_DIR / "TableS19_survival_baselines.csv", index=False, encoding="utf-8-sig")

    plot_df = res[pd.to_numeric(res["holdout_value"], errors="coerce").notna()].copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
        ax.bar(plot_df["model"], plot_df["holdout_value"], color="#4C78A8")
        ax.set_ylim(0.45, max(0.8, plot_df["holdout_value"].max() + 0.05))
        ax.set_ylabel("Holdout C-index")
        ax.set_title("Supplementary survival baseline")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "FigS10_survival_baseline_cindex.png", bbox_inches="tight")
        plt.close(fig)


def utility_counts(y, score, threshold: float, scenario: dict[str, float]) -> dict[str, float]:
    pred_y = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_y, labels=[0, 1]).ravel()
    utility = tp * scenario["benefit_tp"] - fp * scenario["cost_fp"] - fn * scenario["cost_fn"] - tn * scenario["cost_tn"]
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "expected_utility": float(utility),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "positive_prediction_rate": float((tp + fp) / len(y)),
    }


def make_cost_sensitive_utility(valid_pred: pd.DataFrame, test_pred: pd.DataFrame) -> None:
    scenarios = [
        {"scenario": "low_cost_outreach", "benefit_tp": 10, "cost_fp": 1, "cost_fn": 5, "cost_tn": 0},
        {"scenario": "medium_intervention", "benefit_tp": 20, "cost_fp": 5, "cost_fn": 10, "cost_tn": 0},
        {"scenario": "high_cost_program", "benefit_tp": 50, "cost_fp": 20, "cost_fn": 15, "cost_tn": 0},
    ]
    thresholds = np.linspace(0.01, 0.80, 160)
    y_valid = valid_pred["exit_label_t1"].astype(int).to_numpy()
    valid_score = valid_pred["proba_catboost"].to_numpy()
    y_test = test_pred["exit_label_t1"].astype(int).to_numpy()
    test_score = test_pred["proba_catboost"].to_numpy()
    rows = []
    selected_rows = []
    for scenario in scenarios:
        valid_evals = []
        for threshold in thresholds:
            v = utility_counts(y_valid, valid_score, threshold, scenario)
            t = utility_counts(y_test, test_score, threshold, scenario)
            rows.append({**scenario, "partition": "validation", **v})
            rows.append({**scenario, "partition": "test", **t})
            valid_evals.append(v)
                                                                        
                                                                            
        best = max(valid_evals, key=lambda item: item["expected_utility"])
        test_at_best = utility_counts(y_test, test_score, best["threshold"], scenario)
        selected_rows.append(
            {
                **scenario,
                "selection_partition": "validation",
                "selected_threshold": best["threshold"],
                "validation_expected_utility": best["expected_utility"],
                "test_expected_utility": test_at_best["expected_utility"],
                "test_tp": test_at_best["tp"],
                "test_fp": test_at_best["fp"],
                "test_fn": test_at_best["fn"],
                "test_tn": test_at_best["tn"],
                "test_precision": test_at_best["precision"],
                "test_recall": test_at_best["recall"],
                "test_positive_prediction_rate": test_at_best["positive_prediction_rate"],
            }
        )
    pd.DataFrame(rows).to_csv(TAB_DIR / "TableS20_cost_sensitive_threshold_utility.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(TAB_DIR / "TableS20_cost_sensitive_threshold_utility_selected.csv", index=False, encoding="utf-8-sig")

    curve = pd.DataFrame(rows)
    test_curve = curve[curve["partition"] == "test"]
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    for scenario, g in test_curve.groupby("scenario"):
        ax.plot(g["threshold"], g["expected_utility"], label=scenario)
    for row in selected_rows:
        ax.axvline(row["selected_threshold"], linestyle="--", linewidth=0.9, alpha=0.35)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected utility, arbitrary units")
    ax.set_title("Illustrative cost-sensitive threshold utility scenarios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS11_expected_utility_by_threshold.png", bbox_inches="tight")
    plt.close(fig)


def make_transferability_mapping() -> None:
    mapping = [
        {"KLIPS_block": "demographic", "KLIPS_feature": "age_final", "IBM_candidate": "Age", "status": "direct"},
        {"KLIPS_block": "demographic", "KLIPS_feature": "gender", "IBM_candidate": "Gender", "status": "direct"},
        {"KLIPS_block": "job_structure", "KLIPS_feature": "occupation_major", "IBM_candidate": "JobRole", "status": "approximate"},
        {"KLIPS_block": "job_structure", "KLIPS_feature": "tenure_years", "IBM_candidate": "YearsAtCompany", "status": "approximate"},
        {"KLIPS_block": "compensation", "KLIPS_feature": "monthly_wage", "IBM_candidate": "MonthlyIncome", "status": "approximate"},
        {"KLIPS_block": "satisfaction", "KLIPS_feature": "job_sat_*", "IBM_candidate": "JobSatisfaction / EnvironmentSatisfaction", "status": "partial"},
        {"KLIPS_block": "panel_history", "KLIPS_feature": "lagged changes / rolling histories", "IBM_candidate": "", "status": "not available"},
    ]
    pd.DataFrame(mapping).to_csv(TAB_DIR / "TableS21_external_transfer_mapping.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    manifest = {"tasks_completed": []}

    df = read_analysis_base()
    valid_pred, test_pred = load_predictions()

    make_eda(df)
    manifest["tasks_completed"].append("eda_by_exit")

    make_roc_pr_lift_confusion(valid_pred, test_pred)
    manifest["tasks_completed"].append("roc_pr_lift_confusion")

    make_shap_local_and_dependence(df)
    manifest["tasks_completed"].append("shap_local_dependence_interaction")

    make_oversampling_ablation(df)
    manifest["tasks_completed"].append("oversampling_ablation")

    make_survival_baseline(df)
    manifest["tasks_completed"].append("survival_baseline")

    make_cost_sensitive_utility(valid_pred, test_pred)
    manifest["tasks_completed"].append("cost_sensitive_threshold_utility")

    make_transferability_mapping()
    manifest["tasks_completed"].append("transferability_mapping")

    make_tabnet_optional_baseline(df)
    manifest["tasks_completed"].append("tabnet_optional_baseline")

    manifest["outputs"] = {
        "figures_dir": str(FIG_DIR),
        "tables_dir": str(TAB_DIR),
        "models_dir": str(MODEL_DIR),
    }
    with open(LOG_DIR / "additional_analyses_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

