# KLIPS SR Reproducibility Code

This repository contains the analysis code used for the Scientific Reports submission based on the Korean Labor and Income Panel Study public-use files.

## Repository Layout

- `python/`: numbered Python analysis pipeline and helper modules.
- `config/`: expected raw-file inventory and feature/variable mapping metadata.
- `outputs_klips_sr/paper_tables/`: aggregate paper-table source files.
- `outputs_klips_sr/figure_source_data/`: aggregate figure source data and figure image outputs selected for review.
- `outputs_klips_sr/validation_outputs/`: regenerated-artifact validation summary.
- `raw/`: placeholder directory for local KLIPS public-use raw files.

The raw KLIPS files are not redistributed in this public code package. Download the public-use KLIPS Wave 1-27 English files, codebooks, questionnaires, and user guide from the official Korea Labor Institute KLIPS portal, then place the Excel raw files in `raw/` before running the full pipeline.

## Environment

The pipeline was verified with Python 3.11.4. Install the pinned top-level dependencies with:

```bash
pip install -r requirements.txt
```

If a CPU-only environment is used, install the appropriate CPU build of PyTorch before installing the remaining dependencies.

## Reproduction

From the repository root:

```bash
python python/00_run_full_pipeline.py
python python/15_validate_submission_consistency.py
```

The full run reconstructs the analysis base from local KLIPS raw files, trains the primary and supplementary models, regenerates paper tables and figures, and writes validation outputs.

## Main Entry Points

- `python/00_run_full_pipeline.py`: runs stages 01-15 in order.
- `python/01_build_analysis_base.py`: constructs the person-wave analysis base.
- `python/02_train_multimodel.py`: trains primary model baselines.
- `python/03_train_hybrid_bootstrap.py`: trains the hybrid stack and bootstrap intervals.
- `python/04_explainability_and_segments.py`: produces SHAP and subgroup diagnostics.
- `python/05_quality_diagnostics.py`: produces quality diagnostics and main curves.
- `python/06_random_vs_chronological_ablation.py`: runs the random-split ablation.
- `python/07_posthoc_recalibration.py`: runs post hoc calibration diagnostics.
- `python/08_additional_analyses.py`: produces additional supplementary analyses.
- `python/09_person_level_sensitivity.py`: runs person-level sensitivity checks.
- `python/10_calibration_error_metrics.py`: computes calibration-error metrics.
- `python/11_train_period_sensitivity.py`: runs train-period sensitivity checks.
- `python/12_paired_auc_bootstrap.py`: computes paired AUC bootstrap intervals.
- `python/13_feature_block_destination_hours_ablation.py`: runs feature-block, hours, and destination diagnostics.
- `python/14_build_paper_tables.py`: collects manuscript and supplementary source tables.
- `python/15_validate_submission_consistency.py`: validates regenerated analysis artifacts.

## Validation Scope

`python/15_validate_submission_consistency.py` validates regenerated analysis artifacts only. It does not read, create, copy, overwrite, or modify manuscript Word files.

Expected validation anchors include:

- analytical rows: 139,521
- overall exits: 15,167
- model feature count: 138
- CatBoost test PR-AUC: 0.1997
- hybrid stacking test ROC-AUC: 0.7345

## Data Availability Note

The code assumes that the user has obtained the KLIPS public-use files under the data-use terms of the Korea Labor Institute. The `raw/` directory in this repository is intentionally a placeholder and should not contain redistributed raw data unless redistribution is explicitly permitted.
