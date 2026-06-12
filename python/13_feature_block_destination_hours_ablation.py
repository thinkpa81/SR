
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_lib.features import FEATURE_GROUPS, NUMERIC_FEATURES
from pipeline_lib.modeling import ensure_dir, evaluate_binary_classifier, fit_catboost_pipeline, predict_catboost, setup_logging
from pipeline_lib.paths import resolve_project_dir


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

TARGET_COL = "exit_label_t1"

                                           
TRAIN_END_WAVE = 20
VALID_WAVES = (21, 23)
TEST_WAVES = (24, 26)

                                                                                    
MIN_FEATURE_OBS = 20
                                                                              
MIN_DESTINATION_POSITIVES = 100

                                                                    
WAGE_WORKER_STATUS_CODES = [1, 2, 3]
NON_WAGE_WORKER_STATUS_CODES = [4, 5, 6]

GROUP_ORDER = [
    "baseline_harmonised",
    "expanded_raw_questionnaire",
    "nonlinear_interactions",
    "time_aware_panel_features",
]
BLOCKS = {
    "baseline_17": ["baseline_harmonised"],
    "baseline_plus_expanded": ["baseline_harmonised", "expanded_raw_questionnaire"],
    "baseline_plus_expanded_plus_nonlinear": ["baseline_harmonised", "expanded_raw_questionnaire", "nonlinear_interactions"],
    "full_138": GROUP_ORDER,
}
HOURS_TOKENS = [
    "hour",
    "hours",
    "weekly_hours",
    "overtime",
    "gt48",
    "gt52",
    "ge55",
    "work_days",
    "wage_x_hours",
    "hours_x_tenure",
    "hours_change",
    "hours_pct",
    "hours_rolling",
    "hours_percentile",
    "job_sat_hours",
    "fixed_work_hours",
]
METRIC_COLUMNS = ["roc_auc", "pr_auc", "brier", "recall_at_20", "lift_at_20"]


def feature_list_for_groups(groups: list[str], df: pd.DataFrame) -> list[str]:
    features = []
    for group in groups:
        for feature in FEATURE_GROUPS[group]:
            if feature in df.columns and int(df[feature].notna().sum()) >= MIN_FEATURE_OBS:
                features.append(feature)
    return features


def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["wave"] <= TRAIN_END_WAVE].copy()
    valid = df[df["wave"].between(*VALID_WAVES)].copy()
    test = df[df["wave"].between(*TEST_WAVES)].copy()
    return train, valid, test


def evaluate_catboost_feature_set(
    name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict:
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in features if feature not in NUMERIC_FEATURES]
    y_train = train[TARGET_COL].astype(int).to_numpy()
    y_test = test[TARGET_COL].astype(int).to_numpy()
    model = fit_catboost_pipeline(train[features], y_train, numeric, categorical)
    test_prob = predict_catboost(model, test[features], numeric, categorical)
    row = {
        "analysis": name,
        "model": "catboost",
        "feature_count": len(features),
        "train_n": int(len(train)),
        "validation_n": int(len(valid)),
        "test_n": int(len(test)),
    }
    row.update(evaluate_binary_classifier(y_test, test_prob))
    return row


def add_deltas(df: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if reference is None:
        for metric in METRIC_COLUMNS:
            out[f"delta_{metric}_vs_previous"] = out[metric].diff()
        return out
    ref = out[out["analysis"].eq(reference)]
    if ref.empty:
        return out
    ref_row = ref.iloc[0]
    for metric in METRIC_COLUMNS:
        out[f"delta_{metric}_vs_{reference}"] = out[metric] - ref_row[metric]
    return out


def is_hours_feature(feature: str) -> bool:
    lower = feature.lower()
    return any(token in lower for token in HOURS_TOKENS)


def destination_labels(project_dir: Path, analysis: pd.DataFrame, test_predictions: pd.DataFrame) -> pd.DataFrame:
    panel_path = project_dir / "outputs_klips_sr" / "interim" / "panel_master_raw.csv"
    base = analysis[["pid", "wave", "wave_t1", TARGET_COL]].copy()
    if not panel_path.exists():
        base["destination_t1"] = np.where(base[TARGET_COL].eq(1), "missing worker-status detail", "remained wage-employed")
    else:
        panel = pd.read_csv(
            panel_path,
            usecols=lambda column: column in {"pid", "wave", "employee_status_raw", "employment_status_raw"},
        )
        panel["employee_status_raw"] = pd.to_numeric(panel.get("employee_status_raw"), errors="coerce")
        panel["employment_status_raw"] = pd.to_numeric(panel.get("employment_status_raw"), errors="coerce")
        panel["is_wage_worker_t"] = panel["employee_status_raw"].isin(WAGE_WORKER_STATUS_CODES).astype(float)
        panel["is_non_wage_worker_t"] = panel["employee_status_raw"].isin(NON_WAGE_WORKER_STATUS_CODES).astype(float)
        panel["is_employed_t"] = panel["employment_status_raw"].notna().astype(float)
        next_state = panel.rename(
            columns={
                "wave": "wave_t1",
                "is_wage_worker_t": "is_wage_worker_next",
                "is_non_wage_worker_t": "is_non_wage_worker_next",
                "is_employed_t": "is_employed_next",
            }
        )
        base = base.merge(next_state, on=["pid", "wave_t1"], how="left")
        base["destination_t1"] = "remained wage-employed"
        exit_mask = base[TARGET_COL].eq(1)
        base.loc[exit_mask & base["is_non_wage_worker_next"].eq(1), "destination_t1"] = "non-wage employment"
        base.loc[exit_mask & base["is_employed_next"].eq(0), "destination_t1"] = "non-employed or non-wage state"
        base.loc[exit_mask & base["destination_t1"].eq("remained wage-employed"), "destination_t1"] = "missing worker-status detail"
    return test_predictions.merge(base[["pid", "wave", "destination_t1"]], on=["pid", "wave"], how="left")


def destination_tables(project_dir: Path, analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {pred_path}. Run 03_train_hybrid_bootstrap.py first.")
    pred = pd.read_csv(pred_path)
    scored = destination_labels(project_dir, analysis, pred)
    exit_scored = scored[scored[TARGET_COL].eq(1)].copy()
    composition = (
        exit_scored["destination_t1"]
        .fillna("missing worker-status detail")
        .value_counts(dropna=False)
        .rename_axis("destination_t1")
        .reset_index(name="positive_n")
    )
    composition["positive_share"] = composition["positive_n"] / composition["positive_n"].sum()
    composition["note"] = "Coarse destination categories reconstructed from t+1 worker-status flags; finer retirement/unemployment subtypes were not asserted."

    rows = []
    y_all = scored[TARGET_COL].to_numpy(dtype=int)
    score = scored["proba_catboost"].to_numpy(dtype=float)
    for _, row in composition.iterrows():
        destination = row["destination_t1"]
        y_destination = scored["destination_t1"].eq(destination).to_numpy(dtype=int)
        positives = int(y_destination.sum())
        if positives < MIN_DESTINATION_POSITIVES or len(np.unique(y_destination)) < 2:
            rows.append(
                {
                    "destination_t1": destination,
                    "positive_n": positives,
                    "test_n": int(len(scored)),
                    "base_rate": positives / len(scored),
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "note": "Not estimated because the subtype has fewer than 100 positives or lacks class variation.",
                }
            )
            continue
        metrics = evaluate_binary_classifier(y_destination, score)
        rows.append(
            {
                "destination_t1": destination,
                "positive_n": positives,
                "test_n": int(len(scored)),
                "base_rate": positives / len(scored),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "note": "One-vs-rest diagnostic using the primary CatBoost wage-exit score; not a separately trained destination classifier.",
            }
        )
    performance = pd.DataFrame(rows)
    performance["overall_exit_prevalence"] = float(y_all.mean())
    return composition, performance


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Feature-block, destination, and hours-block ablations for the SR resubmission.")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = args.project_dir.resolve()
    output_dir = project_dir / "sr_additional_analyses" / "feature_block_destination_hours_ablation"
    table_dir = project_dir / "outputs_klips_sr" / "tables"
    ensure_dir(output_dir)
    ensure_dir(table_dir)
    setup_logging(output_dir / "logs" / "16_feature_block_destination_hours_ablation.log")

    analysis_path = project_dir / "outputs_klips_sr" / "processed" / "analysis_base_with_label.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis dataset: {analysis_path}. Run 01_build_analysis_base.py first.")
    df = pd.read_csv(analysis_path)
    train, valid, test = split_chronological(df)

                                     
    block_rows = []
    for block_name, groups in BLOCKS.items():
        features = feature_list_for_groups(groups, df)
        row = evaluate_catboost_feature_set(block_name, train, valid, test, features)
        row["feature_groups"] = "+".join(groups)
        block_rows.append(row)
    block_df = add_deltas(pd.DataFrame(block_rows))
    block_df.to_csv(output_dir / "feature_block_incremental_ablation.csv", index=False, encoding="utf-8-sig")
    block_df.to_csv(table_dir / "TableS33_feature_block_incremental_ablation.csv", index=False, encoding="utf-8-sig")

                                                                                
    full_features = feature_list_for_groups(BLOCKS["full_138"], df)
    without_hours = [feature for feature in full_features if not is_hours_feature(feature)]
    hours_rows = [
        evaluate_catboost_feature_set("full_138", train, valid, test, full_features),
        evaluate_catboost_feature_set("full_without_hours_features", train, valid, test, without_hours),
    ]
    hours_df = add_deltas(pd.DataFrame(hours_rows), reference="full_138")
    hours_df["removed_feature_count"] = [0, len(full_features) - len(without_hours)]
    hours_df.to_csv(output_dir / "hours_block_removal_sensitivity.csv", index=False, encoding="utf-8-sig")
    hours_df.to_csv(table_dir / "TableS34_hours_block_removal_sensitivity.csv", index=False, encoding="utf-8-sig")

                                                               
    composition, performance = destination_tables(project_dir, df)
    composition.to_csv(output_dir / "destination_composition_detailed.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output_dir / "destination_stratified_performance.csv", index=False, encoding="utf-8-sig")
    table35 = performance.merge(composition[["destination_t1", "positive_share"]], on="destination_t1", how="left")
    table35.to_csv(table_dir / "TableS35_destination_stratified_diagnostics.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "input_file": str(analysis_path),
        "split": "train waves <=20, validation waves 21-23, test waves 24-26",
        "model": "CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05)",
        "outputs": [
            "feature_block_incremental_ablation.csv",
            "hours_block_removal_sensitivity.csv",
            "destination_composition_detailed.csv",
            "destination_stratified_performance.csv",
            "TableS33_feature_block_incremental_ablation.csv",
            "TableS34_hours_block_removal_sensitivity.csv",
            "TableS35_destination_stratified_diagnostics.csv",
        ],
        "guardrail": "No held-out test labels were used for feature engineering, tuning, calibration, or threshold selection. Destination diagnostics use the primary CatBoost score as a one-vs-rest diagnostic, not as a destination-specific classifier.",
    }
    with open(output_dir / "ablation_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
