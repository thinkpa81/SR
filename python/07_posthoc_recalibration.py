
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline_lib.modeling import (
    apply_calibrators,
    calibration_table,
    choose_threshold_by_f1,
    choose_threshold_for_top_share,
    ensure_dir,
    evaluate_binary_classifier,
    fit_calibrators,
    setup_logging,
    threshold_classification_metrics,
)
from pipeline_lib.paths import resolve_project_dir

DEFAULT_PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

                                                                                    
MANUSCRIPT_MODELS = {"catboost", "hybrid_stack"}


def plot_calibration_variants(y_true, raw_prob, platt_prob, isotonic_prob, title: str, out_path: Path) -> None:
    raw_df = calibration_table(y_true, raw_prob)
    platt_df = calibration_table(y_true, platt_prob)
    iso_df = calibration_table(y_true, isotonic_prob)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.plot(raw_df["predicted"], raw_df["observed"], marker="o", label="Raw")
    plt.plot(platt_df["predicted"], platt_df["observed"], marker="o", label="Platt-style")
    plt.plot(iso_df["predicted"], iso_df["observed"], marker="o", label="Isotonic")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_threshold_rows(model_name: str, method_name: str, y_valid, valid_prob, y_test, test_prob):
    rows = []
    threshold_map = {
        "default_0.5": 0.5,
        "best_f1_valid": choose_threshold_by_f1(y_valid, valid_prob),
        "top10_valid_cutoff": choose_threshold_for_top_share(valid_prob, 0.10),
        "top20_valid_cutoff": choose_threshold_for_top_share(valid_prob, 0.20),
    }
    for threshold_name, threshold_value in threshold_map.items():
        valid_stats = threshold_classification_metrics(y_valid, valid_prob, threshold_value)
        test_stats = threshold_classification_metrics(y_test, test_prob, threshold_value)
        rows.append(
            {
                "model": model_name,
                "calibration_method": method_name,
                "threshold_rule": threshold_name,
                "selected_threshold": float(threshold_value),
                "valid_f1": valid_stats["f1"],
                "valid_precision": valid_stats["precision"],
                "valid_recall": valid_stats["recall"],
                "valid_positive_prediction_rate": valid_stats["positive_prediction_rate"],
                "test_f1": test_stats["f1"],
                "test_precision": test_stats["precision"],
                "test_recall": test_stats["recall"],
                "test_positive_prediction_rate": test_stats["positive_prediction_rate"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc recalibration experiments for KLIPS SR manuscript.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    args = parser.parse_args()

    project_dir = args.project_dir
    output_dir = project_dir / "sr_additional_analyses" / "posthoc_recalibration"
    ensure_dir(output_dir)
    setup_logging(output_dir / "logs" / "09_posthoc_recalibration.log")

    valid_prediction_file = (
        project_dir
        / "sr_additional_analyses"
        / "ablation_random_vs_chronological"
        / "chronological_valid_predictions.csv"
    )
    test_prediction_file = project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv"
    if not valid_prediction_file.exists():
        raise FileNotFoundError(
            "Run 07_random_vs_chronological_ablation.py first so validation predictions are available: "
            f"{valid_prediction_file}"
        )
    if not test_prediction_file.exists():
        raise FileNotFoundError(
            "Run 03_train_hybrid_bootstrap.py first so test predictions are available: "
            f"{test_prediction_file}"
        )

    valid_pred_df = pd.read_csv(valid_prediction_file)
    test_pred_df = pd.read_csv(test_prediction_file)

    y_valid = valid_pred_df["exit_label_t1"].to_numpy(dtype=int)
    y_test = test_pred_df["exit_label_t1"].to_numpy(dtype=int)
    prob_cols = [c for c in test_pred_df.columns if c.startswith("proba_")]
    rows = []
    threshold_rows = []

    combined_out = (
        test_pred_df[["pid", "wave", "exit_label_t1"]].copy()
        if {"pid", "wave", "exit_label_t1"}.issubset(test_pred_df.columns)
        else pd.DataFrame(index=test_pred_df.index)
    )

    for prob_col in prob_cols:
        model_name = prob_col.replace("proba_", "")
        raw_valid = valid_pred_df[prob_col].to_numpy(dtype=float)
        raw_test = test_pred_df[prob_col].to_numpy(dtype=float)
        platt, isotonic = fit_calibrators(raw_valid, y_valid)
        platt_test, isotonic_test = apply_calibrators(raw_test, platt, isotonic)
        platt_valid, isotonic_valid = apply_calibrators(raw_valid, platt, isotonic)

        combined_out[f"{prob_col}_raw"] = raw_test
        combined_out[f"{prob_col}_platt"] = platt_test
        combined_out[f"{prob_col}_isotonic"] = isotonic_test

        method_payloads = [
            ("raw", raw_valid, raw_test),
            ("platt", platt_valid, platt_test),
            ("isotonic", isotonic_valid, isotonic_test),
        ]

        for method_name, valid_probs, test_probs in method_payloads:
            row = {"model": model_name, "calibration_method": method_name}
            row.update(evaluate_binary_classifier(y_test, test_probs))
            rows.append(row)
            threshold_rows.extend(build_threshold_rows(model_name, method_name, y_valid, valid_probs, y_test, test_probs))

        if model_name in MANUSCRIPT_MODELS:
            plot_calibration_variants(
                y_test,
                raw_test,
                platt_test,
                isotonic_test,
                f"Calibration comparison - {model_name}",
                output_dir / f"figure_calibration_{model_name}.png",
            )

    recal_metrics = pd.DataFrame(rows)
    recal_metrics.to_csv(output_dir / "recalibration_test_metrics.csv", index=False, encoding="utf-8-sig")
    combined_out.to_csv(output_dir / "recalibration_test_predictions_all_methods.csv", index=False, encoding="utf-8-sig")

                                                                                   
    best_brier = (
        recal_metrics.sort_values(["model", "brier", "pr_auc"], ascending=[True, True, False])
        .groupby("model", as_index=False)
        .first()
        .rename(columns={"calibration_method": "best_method_by_brier"})
    )
    best_brier.to_csv(output_dir / "recalibration_best_method_summary.csv", index=False, encoding="utf-8-sig")

    manuscript_subset = recal_metrics[recal_metrics["model"].isin(MANUSCRIPT_MODELS)].copy()
    manuscript_subset.to_csv(output_dir / "recalibration_manuscript_subset.csv", index=False, encoding="utf-8-sig")

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(output_dir / "recalibration_threshold_diagnostics.csv", index=False, encoding="utf-8-sig")

    threshold_subset = threshold_df[threshold_df["model"].isin(MANUSCRIPT_MODELS)].copy()
    threshold_subset.to_csv(output_dir / "recalibration_threshold_subset_for_manuscript.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "valid_prediction_file": str(valid_prediction_file),
        "test_prediction_file": str(test_prediction_file),
        "outputs": [
            "recalibration_test_metrics.csv",
            "recalibration_best_method_summary.csv",
            "recalibration_manuscript_subset.csv",
            "recalibration_test_predictions_all_methods.csv",
            "recalibration_threshold_diagnostics.csv",
            "recalibration_threshold_subset_for_manuscript.csv",
            "figure_calibration_catboost.png",
            "figure_calibration_hybrid_stack.png",
        ],
        "note": "Platt-style and isotonic calibrators are fit on the chronological validation partition and evaluated on the held-out chronological test partition. Additional threshold diagnostics are selected on the validation partition to avoid test-set threshold tuning.",
    }
    with open(output_dir / "recalibration_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
