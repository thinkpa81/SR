
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from pipeline_lib.modeling import ensure_dir, setup_logging
from pipeline_lib.paths import resolve_project_dir

DEFAULT_PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

TARGET_COL = "exit_label_t1"


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs: np.ndarray, sigma: np.ndarray) -> float:
    contrast = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)).item() / np.sqrt(np.dot(np.dot(contrast, sigma), contrast.T)).item()
    return float(2 * (1 - stats.norm.cdf(z)))


def delong_roc_test(y_true: np.ndarray, pred_one: np.ndarray, pred_two: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    pred_one = np.asarray(pred_one, dtype=float)
    pred_two = np.asarray(pred_two, dtype=float)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, sigma = fast_delong(preds, label_1_count)
    return float(aucs[0]), float(aucs[1]), calc_pvalue(aucs, sigma)


def resolve_prediction_file(project_dir: Path) -> Path:
    prediction_path = project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv"
    if prediction_path.exists():
        return prediction_path
    raise FileNotFoundError(f"Missing stage3 prediction file: {prediction_path}")


def save_roc_curves(df: pd.DataFrame, y_true: np.ndarray, prob_cols: list[str], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for col in prob_cols:
        fpr, tpr, _ = roc_curve(y_true, df[col].to_numpy())
        score = roc_auc_score(y_true, df[col].to_numpy())
        plt.plot(fpr, tpr, label=f"{col.replace('proba_', '')} (AUC={score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on held-out test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_pr_curves(df: pd.DataFrame, y_true: np.ndarray, prob_cols: list[str], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for col in prob_cols:
        precision, recall, _ = precision_recall_curve(y_true, df[col].to_numpy())
        score = average_precision_score(y_true, df[col].to_numpy())
        plt.plot(recall, precision, label=f"{col.replace('proba_', '')} (AP={score:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision\u2013Recall curves on held-out test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_hyperparameter_table() -> pd.DataFrame:
    rows = [
        {
            "stage": "stage2",
            "model": "logistic",
            "hyperparameters": json.dumps({"solver": "lbfgs", "max_iter": 500, "class_weight": "balanced"}, ensure_ascii=False),
            "tuning_note": "Fixed configuration; no grid/random/Bayesian search implemented.",
        },
        {
            "stage": "stage2",
            "model": "random_forest",
            "hyperparameters": json.dumps({"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "class_weight": "balanced_subsample", "random_state": 42}, ensure_ascii=False),
            "tuning_note": "Fixed configuration from uploaded code.",
        },
        {
            "stage": "stage2",
            "model": "xgboost",
            "hyperparameters": json.dumps({"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary:logistic", "eval_metric": "auc", "random_state": 42}, ensure_ascii=False),
            "tuning_note": "Fixed configuration from uploaded code.",
        },
        {
            "stage": "stage2",
            "model": "catboost",
            "hyperparameters": json.dumps({"iterations": 300, "depth": 6, "learning_rate": 0.05, "loss_function": "Logloss", "eval_metric": "AUC", "random_seed": 42}, ensure_ascii=False),
            "tuning_note": "Fixed configuration from uploaded code.",
        },
        {
            "stage": "stage3",
            "model": "hybrid_meta_logistic",
            "hyperparameters": json.dumps({"solver": "lbfgs", "max_iter": 500, "class_weight": "balanced", "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"}, ensure_ascii=False),
            "tuning_note": "OOF stacking configuration used in stage 3.",
        },
        {
            "stage": "stage4",
            "model": "catboost_retrain_for_shap",
            "hyperparameters": json.dumps({"iterations": 400, "depth": 6, "learning_rate": 0.05, "loss_function": "Logloss", "eval_metric": "AUC", "random_seed": 42, "shap_sample_n": 5000}, ensure_ascii=False),
            "tuning_note": "Retrained explainability model for SHAP and robustness analysis.",
        },
    ]
    return pd.DataFrame(rows)


def build_metric_summary(df: pd.DataFrame, y_true: np.ndarray, prob_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in prob_cols:
        prob = df[col].to_numpy()
        rows.append(
            {
                "model": col.replace("proba_", ""),
                "roc_auc": roc_auc_score(y_true, prob),
                "pr_auc": average_precision_score(y_true, prob),
                "brier": brier_score_loss(y_true, prob),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def build_delong_table(df: pd.DataFrame, y_true: np.ndarray, prob_cols: list[str]) -> pd.DataFrame:
    rows = []
    for i, col_a in enumerate(prob_cols):
        for col_b in prob_cols[i + 1 :]:
            auc_a, auc_b, pvalue = delong_roc_test(y_true, df[col_a].to_numpy(), df[col_b].to_numpy())
            rows.append(
                {
                    "model_a": col_a.replace("proba_", ""),
                    "model_b": col_b.replace("proba_", ""),
                    "auc_a": auc_a,
                    "auc_b": auc_b,
                    "auc_diff": auc_a - auc_b,
                    "p_value": pvalue,
                }
            )
    return pd.DataFrame(rows).sort_values(["p_value", "auc_diff"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SR quality-control artifacts from held-out prediction vectors.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    args = parser.parse_args()

    project_dir = args.project_dir
    output_root = project_dir / "outputs_klips_sr" / "quality_diagnostics"
    ensure_dir(output_root)
    setup_logging(output_root / "logs" / "06_quality_diagnostics.log")

    prediction_file = resolve_prediction_file(project_dir)
    df = pd.read_csv(prediction_file)
    y_true = df[TARGET_COL].to_numpy(dtype=int)
    prob_cols = [c for c in df.columns if c.startswith("proba_")]
    logging.info("Loaded predictions: %s", prediction_file)
    logging.info("Probability columns: %s", prob_cols)

    metric_df = build_metric_summary(df, y_true, prob_cols)
    metric_df.to_csv(output_root / "model_metric_summary.csv", index=False, encoding="utf-8-sig")

    delong_df = build_delong_table(df, y_true, prob_cols)
    delong_df.to_csv(output_root / "pairwise_delong_test.csv", index=False, encoding="utf-8-sig")

    hyper_df = build_hyperparameter_table()
    hyper_df.to_csv(output_root / "hyperparameter_summary.csv", index=False, encoding="utf-8-sig")

    save_roc_curves(df, y_true, prob_cols, output_root / "figure_test_roc_curves.png")
    save_pr_curves(df, y_true, prob_cols, output_root / "figure_test_pr_curves.png")

    manifest = {
        "prediction_source": str(prediction_file),
        "outputs": [
            "model_metric_summary.csv",
            "pairwise_delong_test.csv",
            "hyperparameter_summary.csv",
            "figure_test_roc_curves.png",
            "figure_test_pr_curves.png",
        ],
    }
    with open(output_root / "quality_diagnostics_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logging.info("Quality diagnostics created under: %s", output_root)


if __name__ == "__main__":
    main()
