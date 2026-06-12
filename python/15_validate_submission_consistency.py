
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from pipeline_lib.features import select_model_columns
from pipeline_lib.modeling import ensure_dir
from pipeline_lib.paths import resolve_project_dir


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

                                                               
CHRONOLOGICAL_PARTITIONS = {
    "train": {"waves": (1, 20)},
    "validation": {"waves": (21, 23)},
    "test": {"waves": (24, 26)},
}

REQUIRED_RANDOM_REFIT_OUTPUTS = [
    "ablation_split_comparison_metrics.csv",
    "ablation_test_summary_for_manuscript.csv",
    "chronological_valid_predictions.csv",
    "chronological_test_predictions.csv",
    "random_valid_predictions.csv",
    "random_test_predictions.csv",
    "figure_ablation_roc_auc.png",
    "figure_ablation_pr_auc.png",
    "figure_ablation_brier.png",
    "figure_ablation_lift20.png",
    "FigS3_random_split_roc_auc.png",
    "FigS4_random_split_lift20.png",
]


def first_existing(paths: Iterable[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing {label}. Checked:\n{candidates}")


def path_label(path: Path, project_dir: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def read_dataset(project_dir: Path) -> pd.DataFrame:
    path = first_existing(
        [
            project_dir / "outputs_klips_sr" / "processed" / "analysis_base_with_label.csv",
            project_dir / "outputs" / "analytical_dataset.parquet",
        ],
        "analytical dataset",
    )
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def audit_partition_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, spec in CHRONOLOGICAL_PARTITIONS.items():
        start_wave, end_wave = spec["waves"]
        part = df[df["wave"].between(start_wave, end_wave)]
        n = int(len(part))
        exits = int(part["exit_label_t1"].sum())
        if n == 0:
            raise AssertionError(f"{name} partition is empty.")
        rows.append({"partition": name, "n": n, "exits": exits, "exit_rate": exits / n})
    return pd.DataFrame(rows)


def read_test_metrics(project_dir: Path) -> pd.DataFrame:
    path = first_existing(
        [
            project_dir / "outputs_klips_sr" / "stage3_hybrid_metrics.csv",
            project_dir / "outputs" / "test_metrics.csv",
        ],
        "test metrics",
    )
    metrics = pd.read_csv(path)
    split_col = "split" if "split" in metrics.columns else "dataset" if "dataset" in metrics.columns else None
    if split_col:
        metrics = metrics[metrics[split_col].astype(str).str.lower().eq("test")].copy()
    return metrics


def audit_catboost_pr_auc(metrics: pd.DataFrame) -> pd.DataFrame:
    catboost = metrics[metrics["model"].eq("catboost")]
    if catboost.empty:
        raise AssertionError("CatBoost test metrics row is missing.")
    pr_auc = float(catboost["pr_auc"].iloc[0])
    if not 0 <= pr_auc <= 1:
        raise AssertionError(f"CatBoost PR-AUC is outside [0, 1]: {pr_auc}")
    return pd.DataFrame(
        [
            {
                "model": "catboost",
                "pr_auc": pr_auc,
                "status": "valid_probability_metric",
            }
        ]
    )


def read_calibration_metrics(project_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = first_existing(
        [
            project_dir / "outputs_klips_sr" / "tables" / "TableS29_calibration_error_metrics.csv",
            project_dir / "sr_additional_analyses" / "calibration_error" / "calibration_error_metrics.csv",
            project_dir / "outputs" / "calibration_error_metrics.csv",
        ],
        "calibration error metrics",
    )
    return pd.read_csv(path), path


def audit_catboost_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    catboost = calibration[calibration["model"].eq("catboost")].set_index("calibration_method")
    for method in ["raw", "platt"]:
        if method not in catboost.index:
            raise AssertionError(f"CatBoost {method} calibration row is missing.")
    if float(catboost.loc["raw", "brier"]) > float(catboost.loc["platt", "brier"]):
        raise AssertionError("Raw CatBoost Brier score is worse than Platt CatBoost.")
    if float(catboost.loc["raw", "ece_10bin"]) > float(catboost.loc["platt", "ece_10bin"]):
        raise AssertionError("Raw CatBoost ECE is worse than Platt CatBoost.")
    return catboost[["brier", "ece_10bin", "mce_10bin"]].reset_index()


def audit_threshold_outputs(project_dir: Path) -> pd.DataFrame:
    path = first_existing(
        [
            project_dir / "outputs_klips_sr" / "tables" / "TableS_threshold_confusion_matrix.csv",
            project_dir / "outputs_klips_sr" / "paper_tables" / "appendix_threshold_diagnostics.csv",
        ],
        "threshold diagnostics",
    )
    table = pd.read_csv(path)
    row = table[
        table["model"].eq("catboost")
        & table["threshold_rule"].eq("best_f1_valid")
    ]
    if row.empty:
        raise AssertionError("CatBoost best-F1 validation threshold row is missing.")
    row = row.iloc[0]
    for column in ["selected_threshold", "precision", "recall", "positive_prediction_rate"]:
        observed = float(row[column])
        if not 0 <= observed <= 1:
            raise AssertionError(f"Threshold diagnostic is outside [0, 1] for {column}: {observed}")
    return pd.DataFrame(
        [
            {
                "source_file": path_label(path, project_dir),
                "model": "catboost",
                "threshold_rule": "best_f1_valid",
                "selected_threshold": float(row["selected_threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "positive_prediction_rate": float(row["positive_prediction_rate"]),
            }
        ]
    )


def audit_random_refit_ablation(project_dir: Path) -> pd.DataFrame:
    output_dir = project_dir / "sr_additional_analyses" / "ablation_random_vs_chronological"
    manifest_path = output_dir / "ablation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing random-refit ablation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "random_refit":
        raise AssertionError(f"Random ablation was not regenerated in random_refit mode: {manifest.get('mode')}")

    rows = []
    for name in REQUIRED_RANDOM_REFIT_OUTPUTS:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing random-refit output: {path}")
        rows.append({"output": name, "bytes": int(path.stat().st_size)})

    summary_path = output_dir / "ablation_test_summary_for_manuscript.csv"
    summary = pd.read_csv(summary_path)
    required_models = {"catboost", "hybrid_stack", "logistic", "random_forest", "xgboost"}
    if set(summary["model"]) != required_models:
        raise AssertionError(f"Unexpected random-refit models: {sorted(set(summary['model']))}")
    return pd.DataFrame(rows)


def audit_table_outputs(project_dir: Path) -> pd.DataFrame:
    required = [
        project_dir / "outputs_klips_sr" / "paper_tables" / "table_2_main_model_performance.csv",
        project_dir / "outputs_klips_sr" / "paper_tables" / "appendix_threshold_diagnostics.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS29_calibration_error_metrics.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS30_train_period_hyperparameter_sensitivity.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS31_paired_auc_delta_ci.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS32_optional_deep_tabular_baseline.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS33_feature_block_incremental_ablation.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS34_hours_block_removal_sensitivity.csv",
        project_dir / "outputs_klips_sr" / "tables" / "TableS35_destination_stratified_diagnostics.csv",
        project_dir / "outputs_klips_sr" / "paper_tables" / "appendix_optional_deep_tabular_baseline.csv",
    ]
    rows = []
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing generated table artifact: {path}")
        rows.append({"path": path_label(path, project_dir), "bytes": int(path.stat().st_size)})
    return pd.DataFrame(rows)


def sha256_text(lines: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_reconstruction_manifest(df: pd.DataFrame, include_internal_key_hash: bool) -> dict:
    exits = int(df["exit_label_t1"].sum())

    feature_cols, _, _ = select_model_columns(df)
    manifest = {
        "analytical_rows": int(len(df)),
        "overall_exits": exits,
        "feature_count": int(len(feature_cols)),
        "feature_list_sha256": sha256_text(sorted(feature_cols)),
        "schema_sha256": sha256_text(f"{column}:{df[column].dtype}" for column in sorted(df.columns)),
    }
    if include_internal_key_hash and {"pid", "wave"}.issubset(df.columns):
        keys = df[["pid", "wave"]].astype(str).agg(":".join, axis=1).sort_values()
        manifest["person_wave_key_sha256_internal_check"] = sha256_text(keys)
    return manifest


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_write_csv(df: pd.DataFrame, path: Path, enabled: bool, project_dir: Path) -> str | None:
    if not enabled:
        return None
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path_label(path, project_dir)


def maybe_write_json(path: Path, payload: dict, enabled: bool, project_dir: Path) -> str | None:
    if not enabled:
        return None
    write_json(path, payload)
    return path_label(path, project_dir)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate regenerated analysis artifacts.")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--write-audit-files",
        action="store_true",
        help="Write optional CSV audit files under outputs_klips_sr/validation.",
    )
    parser.add_argument(
        "--include-internal-key-hash",
        action="store_true",
        help="Include a sorted pid:wave hash for private internal checks. Leave unset for public releases.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = args.project_dir.resolve()
    validation_dir = project_dir / "outputs_klips_sr" / "validation"
    ensure_dir(validation_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = read_dataset(project_dir)
    partition = audit_partition_prevalence(df)
    catboost_pr_auc = audit_catboost_pr_auc(read_test_metrics(project_dir))
    calibration, calibration_path = read_calibration_metrics(project_dir)
    catboost_calibration = audit_catboost_calibration(calibration)
    threshold = audit_threshold_outputs(project_dir)
    random_refit = audit_random_refit_ablation(project_dir)
    table_outputs = audit_table_outputs(project_dir)
    reconstruction = build_reconstruction_manifest(df, include_internal_key_hash=args.include_internal_key_hash)

    optional_outputs = [
        output for output in [
            maybe_write_csv(partition, validation_dir / "partition_prevalence_audit.csv", args.write_audit_files, project_dir),
            maybe_write_csv(catboost_pr_auc, validation_dir / "catboost_pr_auc_audit.csv", args.write_audit_files, project_dir),
            maybe_write_csv(catboost_calibration, validation_dir / "catboost_calibration_artifact_audit.csv", args.write_audit_files, project_dir),
            maybe_write_csv(threshold, validation_dir / "threshold_artifact_audit.csv", args.write_audit_files, project_dir),
            maybe_write_csv(random_refit, validation_dir / "random_refit_artifact_audit.csv", args.write_audit_files, project_dir),
            maybe_write_csv(table_outputs, validation_dir / "table_artifact_audit.csv", args.write_audit_files, project_dir),
            maybe_write_json(validation_dir / "reconstruction_manifest.json", reconstruction, args.write_audit_files, project_dir),
        ]
        if output is not None
    ]

    summary = {
        "status": "passed",
        "scope": "regenerated_analysis_artifacts_only",
        "audit_csv_written": bool(args.write_audit_files),
        "calibration_source_file": path_label(calibration_path, project_dir),
        "reconstruction_manifest": reconstruction,
        "outputs": [
            path_label(validation_dir / "submission_validation_summary.json", project_dir),
            *optional_outputs,
        ],
    }
    write_json(validation_dir / "submission_validation_summary.json", summary)
    logging.info("Regenerated artifact validation passed.")


if __name__ == "__main__":
    main()
