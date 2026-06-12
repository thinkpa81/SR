
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from pipeline_lib.modeling import (
    chronological_overlap_audit,
    cluster_bootstrap_ci,
    ensure_dir,
    evaluate_prediction_subset,
    fit_base_and_hybrid,
    load_analysis_base,
    setup_logging,
    split_chronological,
    split_person_disjoint_chronological,
    subset_new_person_rows,
)
from pipeline_lib.paths import resolve_project_dir

DEFAULT_PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

PID_COL = "pid"
TARGET_COL = "exit_label_t1"
KEY_COLS = ["pid", "wave", "exit_label_t1"]                                                     

                                                                                 
CLUSTER_BOOTSTRAP_N = 300
CLUSTER_BOOTSTRAP_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Person-level sensitivity analyses for the KLIPS SR manuscript.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument(
        "--run-person-disjoint-retrain",
        action="store_true",
        help="Retrain all base and stacked models on person-disjoint chronological partitions. This is slow.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir
    output_dir = project_dir / "sr_additional_analyses" / "person_level_sensitivity"
    ensure_dir(output_dir)
    setup_logging(output_dir / "logs" / "09_person_level_sensitivity.log")

    df = load_analysis_base(project_dir)
    chrono_train, chrono_valid, chrono_test = split_chronological(df)

    overlap_df = chronological_overlap_audit(chrono_train, chrono_valid, chrono_test, id_col=PID_COL)
    overlap_df.to_csv(output_dir / "person_overlap_audit.csv", index=False, encoding="utf-8-sig")

    core_prediction_file = project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv"
    if not core_prediction_file.exists():
        raise FileNotFoundError(
            "Run 03_train_hybrid_bootstrap.py first so chronological test predictions are available: "
            f"{core_prediction_file}"
        )
    core_test_pred = pd.read_csv(core_prediction_file)

                                                                                      
    new_person_test = subset_new_person_rows(chrono_train, chrono_valid, chrono_test, id_col=PID_COL)
    new_person_pred = core_test_pred.copy()
    if not new_person_test.empty and set(KEY_COLS).issubset(new_person_test.columns) and set(KEY_COLS).issubset(core_test_pred.columns):
        new_person_pred = core_test_pred.merge(
            new_person_test[KEY_COLS],
            on=KEY_COLS,
            how="inner",
        )
    prob_cols = [c for c in core_test_pred.columns if c.startswith("proba_")]
    new_person_metrics = evaluate_prediction_subset(new_person_pred, prob_cols, y_col=TARGET_COL)
    new_person_metrics["subset"] = "new_person_only_from_core_test"
    new_person_metrics.to_csv(output_dir / "new_person_only_test_metrics.csv", index=False, encoding="utf-8-sig")

                                                                                   
    disjoint_train, disjoint_valid, disjoint_test = split_person_disjoint_chronological(df, id_col=PID_COL)
    if args.run_person_disjoint_retrain and min(len(disjoint_train), len(disjoint_valid), len(disjoint_test)) > 0:
        disjoint_metrics, _, disjoint_test_pred = fit_base_and_hybrid(disjoint_train, disjoint_valid, disjoint_test, "person_disjoint_chronological")
        disjoint_metrics = disjoint_metrics[disjoint_metrics["dataset"] == "test"].copy()
        disjoint_metrics.to_csv(output_dir / "person_disjoint_test_metrics.csv", index=False, encoding="utf-8-sig")
        disjoint_test_pred.to_csv(output_dir / "person_disjoint_test_predictions.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            [{
                "note": (
                    "Person-disjoint retraining was not run in the default fast sensitivity path. "
                    "Use --run-person-disjoint-retrain to compute this optional, slow check."
                )
                if not args.run_person_disjoint_retrain
                else "One or more person-disjoint chronological partitions are empty; metrics were not computed.",
                "train_n": len(disjoint_train),
                "valid_n": len(disjoint_valid),
                "test_n": len(disjoint_test),
                "run_person_disjoint_retrain": bool(args.run_person_disjoint_retrain),
            }]
        ).to_csv(output_dir / "person_disjoint_test_metrics.csv", index=False, encoding="utf-8-sig")

                                                                      
    cluster_rows = []
    cluster_input = core_test_pred
    if set(KEY_COLS).issubset(chrono_test.columns) and set(KEY_COLS).issubset(core_test_pred.columns):
        cluster_input = chrono_test[KEY_COLS].merge(
            core_test_pred,
            on=KEY_COLS,
            how="inner",
        )
    if PID_COL in cluster_input.columns:
        y_true = cluster_input[TARGET_COL].to_numpy(dtype=int)
        groups = cluster_input[PID_COL].to_numpy()
        for prob_col in prob_cols:
            probs = cluster_input[prob_col].to_numpy(dtype=float)
            for metric_name, metric_fn in [
                ("roc_auc", roc_auc_score),
                ("pr_auc", average_precision_score),
                ("brier", brier_score_loss),
            ]:
                mean_val, ci_low, ci_high = cluster_bootstrap_ci(
                    y_true=y_true,
                    y_prob=probs,
                    groups=groups,
                    metric_fn=metric_fn,
                    n_boot=CLUSTER_BOOTSTRAP_N,
                    seed=CLUSTER_BOOTSTRAP_SEED,
                )
                cluster_rows.append(
                    {
                        "model": prob_col.replace("proba_", ""),
                        "metric": metric_name,
                        "bootstrap_mean": mean_val,
                        "ci_2.5": ci_low,
                        "ci_97.5": ci_high,
                    }
                )
    pd.DataFrame(cluster_rows).to_csv(output_dir / "cluster_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "outputs": [
            "person_overlap_audit.csv",
            "new_person_only_test_metrics.csv",
            "person_disjoint_test_metrics.csv",
            "person_disjoint_test_predictions.csv",
            "cluster_bootstrap_ci.csv",
        ],
        "core_prediction_file": str(core_prediction_file),
        "run_person_disjoint_retrain": bool(args.run_person_disjoint_retrain),
        "note": "This script provides quality-control sensitivity checks for person overlap, new-person-only testing, conservative person-disjoint chronological testing, and person-clustered bootstrap uncertainty.",
    }
    with open(output_dir / "person_level_sensitivity_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
