
from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline_lib.modeling import ensure_dir, setup_logging
from pipeline_lib.paths import resolve_project_dir


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)
MODEL_ORDER = ["logistic", "random_forest", "xgboost", "catboost", "hybrid_stack"]


def model_probability_columns(df: pd.DataFrame) -> list[str]:
    order = {f"proba_{name}": i for i, name in enumerate(MODEL_ORDER)}
    columns = [c for c in df.columns if c.startswith("proba_") and not c.endswith(("_raw", "_platt", "_isotonic"))]
    return sorted(columns, key=lambda c: order.get(c, len(order)))


def paired_auc_delta_ci(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)

    observed_a = float(roc_auc_score(y_true, prob_a))
    observed_b = float(roc_auc_score(y_true, prob_b))
    observed_delta = observed_a - observed_b

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if np.unique(y_boot).size < 2:
            continue
        deltas.append(roc_auc_score(y_boot, prob_a[idx]) - roc_auc_score(y_boot, prob_b[idx]))

    if not deltas:
        return observed_a, observed_b, observed_delta, np.nan, np.nan

    deltas = np.asarray(deltas, dtype=float)
    return (
        observed_a,
        observed_b,
        observed_delta,
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def build_pairwise_table(pred_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    y_true = pred_df["exit_label_t1"].to_numpy(dtype=int)
    columns = model_probability_columns(pred_df)
    rows = []
    for col_a, col_b in itertools.combinations(columns, 2):
        auc_a, auc_b, delta, lo, hi = paired_auc_delta_ci(
            y_true,
            pred_df[col_a].to_numpy(dtype=float),
            pred_df[col_b].to_numpy(dtype=float),
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "model_a": col_a.replace("proba_", ""),
                "model_b": col_b.replace("proba_", ""),
                "auc_a": auc_a,
                "auc_b": auc_b,
                "auc_diff_a_minus_b": delta,
                "ci_2.5": lo,
                "ci_97.5": hi,
                "bootstrap_n": n_boot,
                "method": "paired nonparametric bootstrap on held-out test observations",
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap delta-AUC intervals.")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = args.project_dir.resolve()
    output_dir = project_dir / "sr_additional_analyses" / "paired_auc_bootstrap"
    table_dir = project_dir / "outputs_klips_sr" / "tables"
    ensure_dir(output_dir)
    ensure_dir(table_dir)
    setup_logging(output_dir / "logs" / "14_paired_auc_bootstrap.log")

    prediction_path = project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing {prediction_path}. Run 03_train_hybrid_bootstrap.py first.")

    pred_df = pd.read_csv(prediction_path)
    result = build_pairwise_table(pred_df, n_boot=args.bootstrap_n, seed=args.seed)

    result_path = output_dir / "paired_auc_delta_ci.csv"
    table_path = table_dir / "TableS31_paired_auc_delta_ci.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    result.to_csv(table_path, index=False, encoding="utf-8-sig")

    manifest = {
        "test_predictions": str(prediction_path),
        "bootstrap_n": args.bootstrap_n,
        "seed": args.seed,
        "outputs": [str(result_path), str(table_path)],
    }
    with open(output_dir / "paired_auc_bootstrap_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
