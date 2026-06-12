
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from pipeline_lib.modeling import apply_calibrators, ensure_dir, fit_calibrators, setup_logging
from pipeline_lib.paths import resolve_project_dir


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)
MODEL_ORDER = ["logistic", "random_forest", "xgboost", "catboost", "hybrid_stack"]
CALIBRATION_METHODS = ["raw", "platt", "isotonic"]


def quantile_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    finite = np.isfinite(y_prob)
    y_true = y_true[finite]
    y_prob = y_prob[finite]

    if y_true.size == 0:
        empty = pd.DataFrame(columns=["bin", "n", "mean_predicted", "observed_rate", "absolute_error"])
        return np.nan, np.nan, empty

    order = np.argsort(y_prob)
    bins = np.array_split(order, min(n_bins, y_true.size))
    rows = []
    for bin_index, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue
        mean_predicted = float(y_prob[idx].mean())
        observed_rate = float(y_true[idx].mean())
        rows.append(
            {
                "bin": bin_index,
                "n": int(idx.size),
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "absolute_error": abs(mean_predicted - observed_rate),
            }
        )

    bin_df = pd.DataFrame(rows)
    weighted_error = (bin_df["absolute_error"] * bin_df["n"]).sum() / bin_df["n"].sum()
    max_error = bin_df["absolute_error"].max()
    return float(weighted_error), float(max_error), bin_df


def prediction_columns(df: pd.DataFrame) -> list[str]:
    columns = [c for c in df.columns if c.startswith("proba_")]
    clean = [c for c in columns if not c.endswith(("_raw", "_platt", "_isotonic"))]
    order = {f"proba_{name}": i for i, name in enumerate(MODEL_ORDER)}
    return sorted(clean, key=lambda c: order.get(c, len(order)))


def calibrated_test_probabilities(valid_df: pd.DataFrame, test_df: pd.DataFrame, prob_col: str) -> dict[str, np.ndarray]:
    y_valid = valid_df["exit_label_t1"].to_numpy(dtype=int)
    valid_prob = valid_df[prob_col].to_numpy(dtype=float)
    test_prob = test_df[prob_col].to_numpy(dtype=float)
    platt, isotonic = fit_calibrators(valid_prob, y_valid)
    platt_prob, isotonic_prob = apply_calibrators(test_prob, platt, isotonic)
    return {"raw": test_prob, "platt": platt_prob, "isotonic": isotonic_prob}


def metric_row(model: str, method: str, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> tuple[dict, pd.DataFrame]:
    ece, mce, bin_df = quantile_calibration_error(y_true, y_prob, n_bins=n_bins)
    row = {
        "model": model,
        "calibration_method": method,
        "n": int(len(y_true)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        f"ece_{n_bins}bin": ece,
        f"mce_{n_bins}bin": mce,
    }
    bin_df.insert(0, "calibration_method", method)
    bin_df.insert(0, "model", model)
    return row, bin_df


def build_calibration_error_tables(valid_df: pd.DataFrame, test_df: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_test = test_df["exit_label_t1"].to_numpy(dtype=int)
    rows = []
    bin_tables = []

    for prob_col in prediction_columns(test_df):
        if prob_col not in valid_df.columns:
            logging.warning("Validation predictions missing for %s; skipped.", prob_col)
            continue
        model = prob_col.replace("proba_", "")
        for method, probabilities in calibrated_test_probabilities(valid_df, test_df, prob_col).items():
            row, bin_df = metric_row(model, method, y_test, probabilities, n_bins)
            rows.append(row)
            bin_tables.append(bin_df)

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df["model_order"] = metrics_df["model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
        metrics_df["method_order"] = metrics_df["calibration_method"].map({name: i for i, name in enumerate(CALIBRATION_METHODS)})
        metrics_df = metrics_df.sort_values(["model_order", "method_order"]).drop(columns=["model_order", "method_order"])

    bins_df = pd.concat(bin_tables, ignore_index=True) if bin_tables else pd.DataFrame()
    return metrics_df, bins_df


def required_file(path: Path, producer: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {producer} first.")
    return path


def first_existing(paths: Iterable[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing {label}. Checked:\n{candidates}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute quantile-bin ECE and MCE for calibrated test predictions.")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = args.project_dir.resolve()
    output_dir = project_dir / "sr_additional_analyses" / "calibration_error"
    table_dir = project_dir / "outputs_klips_sr" / "tables"
    ensure_dir(output_dir)
    ensure_dir(table_dir)
    setup_logging(output_dir / "logs" / "12_calibration_error_metrics.log")

    valid_path = required_file(
        first_existing(
            [
                project_dir / "outputs_klips_sr" / "stage3_valid_predictions_with_hybrid.csv",
                project_dir / "sr_additional_analyses" / "ablation_random_vs_chronological" / "chronological_valid_predictions.csv",
            ],
            "validation predictions",
        ),
        "03_train_hybrid_bootstrap.py",
    )
    test_path = required_file(
        project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv",
        "03_train_hybrid_bootstrap.py",
    )

    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    metrics_df, bins_df = build_calibration_error_tables(valid_df, test_df, args.n_bins)

    metrics_path = output_dir / "calibration_error_metrics.csv"
    bins_path = output_dir / "calibration_error_bins.csv"
    table_path = table_dir / "TableS29_calibration_error_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(table_path, index=False, encoding="utf-8-sig")
    bins_df.to_csv(bins_path, index=False, encoding="utf-8-sig")

    manifest = {
        "valid_predictions": str(valid_path),
        "test_predictions": str(test_path),
        "n_bins": args.n_bins,
        "outputs": [str(metrics_path), str(bins_path), str(table_path)],
        "definition": "ECE is the sample-weighted mean absolute difference between observed event rates and mean predicted probabilities across quantile bins; MCE is the largest bin-level absolute difference.",
    }
    with open(output_dir / "calibration_error_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
