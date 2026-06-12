
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PipelineStep:
    number: int
    script: str
    purpose: str
    required_outputs: tuple[str, ...] = ()
    optional_outputs: tuple[str, ...] = ()
    args: tuple[str, ...] = ()


PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        1,
        "01_build_analysis_base.py",
        "Build KLIPS analysis base, wage-worker risk set, labels, feature manifest and run summary.",
        (
            "outputs_klips_sr/processed/analysis_base_with_label.csv",
            "outputs_klips_sr/run_summary.json",
            "outputs_klips_sr/feature_engineering_manifest.csv",
        ),
    ),
    PipelineStep(
        2,
        "02_train_multimodel.py",
        "Train primary single-model baselines and save model metrics.",
        (
            "outputs_klips_sr/stage2_model_metrics.csv",
            "outputs_klips_sr/stage2_summary.json",
        ),
    ),
    PipelineStep(
        3,
        "03_train_hybrid_bootstrap.py",
        "Train hybrid stack, save validation/test predictions, bootstrap CIs and calibration curves.",
        (
            "outputs_klips_sr/stage3_hybrid_metrics.csv",
            "outputs_klips_sr/stage3_bootstrap_ci.csv",
            "outputs_klips_sr/stage3_test_predictions_with_hybrid.csv",
            "outputs_klips_sr/stage3_valid_predictions_with_hybrid.csv",
            "outputs_klips_sr/stage3_summary.json",
        ),
    ),
    PipelineStep(
        4,
        "04_explainability_and_segments.py",
        "Generate CatBoost SHAP, subgroup performance and robustness diagnostics.",
        (
            "outputs_klips_sr/stage4_catboost_shap_importance.csv",
            "outputs_klips_sr/stage4_segment_performance.csv",
        ),
    ),
    PipelineStep(
        5,
        "05_quality_diagnostics.py",
        "Generate quality diagnostics, ROC/PR/lift plots and supplementary outputs.",
    ),
    PipelineStep(
        6,
        "06_random_vs_chronological_ablation.py",
        "Generate chronological-vs-random split comparison outputs.",
        (
            "sr_additional_analyses/ablation_random_vs_chronological/ablation_split_comparison_metrics.csv",
            "sr_additional_analyses/ablation_random_vs_chronological/chronological_valid_predictions.csv",
            "sr_additional_analyses/ablation_random_vs_chronological/chronological_test_predictions.csv",
            "sr_additional_analyses/ablation_random_vs_chronological/random_valid_predictions.csv",
            "sr_additional_analyses/ablation_random_vs_chronological/random_test_predictions.csv",
            "sr_additional_analyses/ablation_random_vs_chronological/FigS3_random_split_roc_auc.png",
            "sr_additional_analyses/ablation_random_vs_chronological/FigS4_random_split_lift20.png",
            "sr_additional_analyses/ablation_random_vs_chronological/ablation_manifest.json",
        ),
        args=("--run-random-refit",),
    ),
    PipelineStep(
        7,
        "07_posthoc_recalibration.py",
        "Generate post-hoc calibration and threshold diagnostics.",
        (
            "sr_additional_analyses/posthoc_recalibration/recalibration_threshold_diagnostics.csv",
        ),
    ),
    PipelineStep(
        8,
        "08_additional_analyses.py",
        "Generate additional figures and diagnostic tables.",
        (
            "outputs_klips_sr/tables/TableS32_optional_deep_tabular_baseline.csv",
        ),
    ),
    PipelineStep(
        9,
        "09_person_level_sensitivity.py",
        "Generate person-overlap, new-person-only and cluster-bootstrap sensitivity outputs.",
        (
            "sr_additional_analyses/person_level_sensitivity/person_overlap_audit.csv",
            "sr_additional_analyses/person_level_sensitivity/new_person_only_test_metrics.csv",
        ),
    ),
    PipelineStep(
        10,
        "10_calibration_error_metrics.py",
        "Generate ECE/MCE and calibration-error metrics.",
        (
            "outputs_klips_sr/tables/TableS29_calibration_error_metrics.csv",
        ),
    ),
    PipelineStep(
        11,
        "11_train_period_sensitivity.py",
        "Generate fixed-configuration train-period sensitivity diagnostics.",
        (
            "outputs_klips_sr/tables/TableS30_train_period_hyperparameter_sensitivity.csv",
        ),
    ),
    PipelineStep(
        12,
        "12_paired_auc_bootstrap.py",
        "Generate paired bootstrap delta-AUC intervals.",
        (
            "outputs_klips_sr/tables/TableS31_paired_auc_delta_ci.csv",
        ),
    ),
    PipelineStep(
        13,
        "13_feature_block_destination_hours_ablation.py",
        "Generate S33-S35: feature-block ablation, hours-feature sensitivity and destination diagnostics.",
        (
            "outputs_klips_sr/tables/TableS33_feature_block_incremental_ablation.csv",
            "outputs_klips_sr/tables/TableS34_hours_block_removal_sensitivity.csv",
            "outputs_klips_sr/tables/TableS35_destination_stratified_diagnostics.csv",
        ),
    ),
    PipelineStep(
        14,
        "14_build_paper_tables.py",
        "Collect generated outputs into paper_tables CSV/XLSX files.",
        (
            "outputs_klips_sr/paper_tables/table_1_dataset_summary.csv",
            "outputs_klips_sr/paper_tables/table_2_main_model_performance.csv",
            "outputs_klips_sr/paper_tables/table_3_bootstrap_ci.csv",
            "outputs_klips_sr/paper_tables/appendix_optional_deep_tabular_baseline.csv",
            "outputs_klips_sr/paper_tables/appendix_feature_block_incremental_ablation.csv",
            "outputs_klips_sr/paper_tables/appendix_hours_block_removal_sensitivity.csv",
            "outputs_klips_sr/paper_tables/appendix_destination_stratified_diagnostics.csv",
        ),
    ),
    PipelineStep(
        15,
        "15_validate_submission_consistency.py",
        "Validate regenerated analysis artifacts.",
        (
            "outputs_klips_sr/validation/submission_validation_summary.json",
        ),
    ),
]


def configure_logging(project_dir: Path) -> Path:
    log_dir = project_dir / "outputs_klips_sr" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "00_run_full_pipeline.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def default_project_dir() -> Path:
                                                                  
    return Path(__file__).resolve().parent.parent


def default_scripts_dir(project_dir: Path) -> Path:
    candidate = project_dir / "python"
    return candidate if candidate.exists() else Path(__file__).resolve().parent


def build_subprocess_env(scripts_dir: Path, project_dir: Path) -> dict[str, str]:
    env = os.environ.copy()

    path_entries = [
        str(scripts_dir),
        str(project_dir),
        str(project_dir / "src"),
    ]

    existing = env.get("PYTHONPATH", "")
    if existing:
        path_entries.append(existing)

                                               
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in path_entries:
        if entry and entry not in seen:
            seen.add(entry)
            deduped.append(entry)

    env["PYTHONPATH"] = os.pathsep.join(deduped)
    env["KLIPS_PROJECT_DIR"] = str(project_dir)
    env["KLIPS_SCRIPTS_DIR"] = str(scripts_dir)

    return env


def relative_missing(project_dir: Path, rel_paths: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for rel_path in rel_paths:
        if not (project_dir / rel_path).exists():
            missing.append(rel_path)
    return missing


def validate_environment(project_dir: Path, scripts_dir: Path) -> None:
    logging.info("Project dir: %s", project_dir)
    logging.info("Scripts dir: %s", scripts_dir)
    logging.info("Python executable: %s", sys.executable)

    if not scripts_dir.exists():
        raise FileNotFoundError(f"Scripts directory not found: {scripts_dir}")

    pipeline_lib_dir = scripts_dir / "pipeline_lib"
    required_lib_files = [
        pipeline_lib_dir / "__init__.py",
        pipeline_lib_dir / "features.py",
        pipeline_lib_dir / "modeling.py",
        pipeline_lib_dir / "paths.py",
    ]

    missing_lib = [path for path in required_lib_files if not path.exists()]
    if missing_lib:
        missing_text = "\n".join(str(path) for path in missing_lib)
        raise FileNotFoundError(
            "pipeline_lib package is incomplete or missing. Required files not found:\n"
            f"{missing_text}\n\n"
            "Expected location:\n"
            f"{pipeline_lib_dir}"
        )

    missing_scripts = [step.script for step in PIPELINE_STEPS if not (scripts_dir / step.script).exists()]
    if missing_scripts:
        raise FileNotFoundError(
            "One or more pipeline scripts are missing from the scripts directory:\n"
            + "\n".join(missing_scripts)
        )


def step_index_from_resume_arg(resume_from: str | None) -> int:
    if not resume_from:
        return 0

    value = resume_from.strip()
    if value.isdigit():
        number = int(value)
        for i, step in enumerate(PIPELINE_STEPS):
            if step.number == number:
                return i
        raise ValueError(f"--resume-from step number not found: {number}")

    for i, step in enumerate(PIPELINE_STEPS):
        if step.script == value or step.script.startswith(value):
            return i

    raise ValueError(f"--resume-from does not match a step number or script name: {resume_from}")


def should_stop_after(step: PipelineStep, stop_after: str | None) -> bool:
    if not stop_after:
        return False
    value = stop_after.strip()
    if value.isdigit():
        return step.number == int(value)
    return step.script == value or step.script.startswith(value)


def run_step(
    step: PipelineStep,
    project_dir: Path,
    scripts_dir: Path,
    python_executable: str,
    skip_output_check: bool = False,
) -> dict:
    script_path = scripts_dir / step.script
    command = [python_executable, str(script_path), *step.args]

    logging.info("=" * 92)
    logging.info("STEP %02d | %s", step.number, step.script)
    logging.info("Purpose: %s", step.purpose)
    logging.info("Command: %s", " ".join(command))

    start = time.time()
    env = build_subprocess_env(scripts_dir=scripts_dir, project_dir=project_dir)

    result = subprocess.run(
        command,
        cwd=str(scripts_dir),
        env=env,
        text=True,
    )

    elapsed = round(time.time() - start, 2)
    record = {
        "number": step.number,
        "script": step.script,
        "purpose": step.purpose,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "required_outputs": list(step.required_outputs),
        "optional_outputs": list(step.optional_outputs),
        "missing_required_outputs": [],
        "missing_optional_outputs": [],
    }

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed at step {step.number:02d} ({step.script}) with return code {result.returncode}."
        )

    if not skip_output_check:
        missing_required = relative_missing(project_dir, step.required_outputs)
        missing_optional = relative_missing(project_dir, step.optional_outputs)
        record["missing_required_outputs"] = missing_required
        record["missing_optional_outputs"] = missing_optional

        if missing_required:
            raise FileNotFoundError(
                f"Step {step.number:02d} completed but required outputs are missing:\n"
                + "\n".join(missing_required)
            )
        if missing_optional:
            logging.warning(
                "Step %02d completed but optional outputs are missing: %s",
                step.number,
                missing_optional,
            )

    logging.info("STEP %02d completed in %.2f seconds.", step.number, elapsed)
    return record


def write_manifest(project_dir: Path, records: list[dict], status: str, error: str | None = None) -> Path:
    log_dir = project_dir / "outputs_klips_sr" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "00_pipeline_run_manifest.json"

    manifest = {
        "status": status,
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "project_dir": str(project_dir),
        "scripts_dir": str(project_dir / "python"),
        "steps": records,
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KLIPS SR pipeline scripts 01 through 15 in dependency-safe order.")
    parser.add_argument("--project-dir", type=Path, default=None, help="Project root directory. Defaults to parent of this script directory.")
    parser.add_argument("--scripts-dir", type=Path, default=None, help="Directory containing numbered pipeline scripts.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for child scripts.")
    parser.add_argument("--resume-from", default=None, help="Resume from step number, e.g. 13, or script name prefix.")
    parser.add_argument("--stop-after", default=None, help="Stop after step number or script name prefix.")
    parser.add_argument("--skip-output-check", action="store_true", help="Run scripts without checking required output files.")
    args = parser.parse_args()

    project_dir = (args.project_dir or default_project_dir()).resolve()
    scripts_dir = (args.scripts_dir or default_scripts_dir(project_dir)).resolve()

    log_path = configure_logging(project_dir)

    records: list[dict] = []
    try:
        logging.info("KLIPS SR full pipeline runner started.")
        logging.info("Log path: %s", log_path)

        validate_environment(project_dir=project_dir, scripts_dir=scripts_dir)

        start_index = step_index_from_resume_arg(args.resume_from)
        selected_steps = PIPELINE_STEPS[start_index:]

        logging.info(
            "Execution order: %s",
            " -> ".join(f"{step.number:02d}:{step.script}" for step in selected_steps),
        )

        for step in selected_steps:
            record = run_step(
                step=step,
                project_dir=project_dir,
                scripts_dir=scripts_dir,
                python_executable=args.python_executable,
                skip_output_check=args.skip_output_check,
            )
            records.append(record)

            if should_stop_after(step, args.stop_after):
                logging.info("Stopped after requested step: %02d %s", step.number, step.script)
                break

        manifest_path = write_manifest(project_dir, records, status="completed")
        logging.info("=" * 92)
        logging.info("KLIPS SR full pipeline completed.")
        logging.info("Run manifest saved to: %s", manifest_path)

    except Exception as exc:
        manifest_path = write_manifest(project_dir, records, status="failed", error=str(exc))
        logging.exception("KLIPS SR pipeline failed. Run manifest saved to: %s", manifest_path)
        raise


if __name__ == "__main__":
    main()
