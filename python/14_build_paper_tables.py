
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_lib.modeling import setup_logging
from pipeline_lib.paths import resolve_project_dir

PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
TABLE_DIR = OUTPUT_DIR / "paper_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

setup_logging(OUTPUT_DIR / "logs" / "klips_stage5_paper_tables.log")
logger = logging.getLogger(__name__)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_table(df: pd.DataFrame, base_name: str) -> None:
    if df.empty:
        logger.warning("Skip empty table: %s", base_name)
        return
    csv_path = TABLE_DIR / f"{base_name}.csv"
    xlsx_path = TABLE_DIR / f"{base_name}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    logger.info("Saved table: %s", base_name)


def format_metric_cols(df: pd.DataFrame, metric_cols: list[str], ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for column in metric_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(ndigits)
    return out


def build_table_1_dataset_summary() -> pd.DataFrame:
    stage1_summary = read_json_if_exists(OUTPUT_DIR / "run_summary.json")
    stage2_summary = read_json_if_exists(OUTPUT_DIR / "stage2_summary.json")
    stage3_summary = read_json_if_exists(OUTPUT_DIR / "stage3_summary.json")
    rows = [
        {"Item": "Initial panel master", "Value": stage1_summary.get("initial_panel_master_n"), "Note": "Before risk-set restrictions"},
        {"Item": "Outside wage-employment risk set", "Value": stage1_summary.get("outside_wage_risk_set_n"), "Note": "Excluded at baseline t"},
        {"Item": "No valid t+1 follow-up", "Value": stage1_summary.get("no_valid_t1_followup_n"), "Note": "Excluded from wage-worker risk set"},
        {"Item": "Analytical sample", "Value": stage1_summary.get("analysis_base_n", stage1_summary.get("analysis_base_shape", [None])[0]), "Note": "Wage-employment observations with exit_label_t1"},
        {"Item": "Training partition", "Value": stage3_summary.get("train_shape", stage1_summary.get("chronological_train_shape", [None]))[0], "Note": "Waves 1-20 (1998-2017)"},
        {"Item": "Validation partition", "Value": stage3_summary.get("valid_shape", stage1_summary.get("chronological_valid_shape", [None]))[0], "Note": "Waves 21-23 (2018-2020)"},
        {"Item": "Held-out test partition", "Value": stage3_summary.get("test_shape", stage1_summary.get("chronological_test_shape", [None]))[0], "Note": "Waves 24-26 (2021-2023)"},
        {"Item": "Overall exit rate", "Value": stage1_summary.get("analysis_base_exit_rate"), "Note": "Proportion exiting wage employment at t+1"},
        {"Item": "Final model features", "Value": stage2_summary.get("feature_count"), "Note": "After excluding features with fewer than 20 observed values"},
    ]
    df = pd.DataFrame(rows)
    if "Value" in df.columns:
        df["Value"] = df["Value"].apply(lambda v: round(v, 4) if isinstance(v, float) else v)
    return df


def build_table_1b_feature_space_summary() -> pd.DataFrame:
    manifest = read_csv_if_exists(OUTPUT_DIR / "feature_engineering_manifest.csv")
    stage2_summary = read_json_if_exists(OUTPUT_DIR / "stage2_summary.json")
    rows = [
        {
            "Feature block": "Candidate feature definitions",
            "Count": int(len(manifest)) if not manifest.empty else np.nan,
            "Note": "Raw questionnaire, harmonised, nonlinear, and panel-history features defined in code",
        },
        {
            "Feature block": "Final model features",
            "Count": stage2_summary.get("feature_count"),
            "Note": "Features used by the modelling scripts after the low-observation screen",
        },
    ]
    if not manifest.empty and "feature_group" in manifest.columns:
        group_counts = manifest.groupby("feature_group").size().reset_index(name="Count")
        for _, row in group_counts.iterrows():
            rows.append(
                {
                    "Feature block": row["feature_group"],
                    "Count": int(row["Count"]),
                    "Note": "Candidate definitions in this feature-engineering block",
                }
            )
    return pd.DataFrame(rows)


def build_table_2_main_model_performance() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_hybrid_metrics.csv")
    if df.empty:
        return df
    df = df[df["split"] == "test"].copy()
    metric_cols = ["roc_auc", "pr_auc", "brier", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    order = {"logistic": 1, "random_forest": 2, "xgboost": 3, "catboost": 4, "hybrid_stack": 5}
    df["model_order"] = df["model"].map(order)
    df = df.sort_values("model_order").drop(columns=["model_order", "split", "f1", "recall_at_10", "lift_at_10"], errors="ignore")
    return df.rename(
        columns={
            "model": "Model",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "brier": "Brier score",
            "recall_at_20": "Recall@20",
            "lift_at_20": "Lift@20",
        }
    )[["Model", "ROC-AUC", "PR-AUC", "Brier score", "Recall@20", "Lift@20"]]


def build_table_3_bootstrap_ci() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_bootstrap_ci.csv")
    if df.empty:
        return df
    df = format_metric_cols(df, ["bootstrap_mean", "ci_2.5", "ci_97.5"])
    df = df[df["metric"].isin(["roc_auc", "pr_auc", "brier"])].copy()
    df["95% CI"] = df.apply(
        lambda row: f"[{row['ci_2.5']:.4f}, {row['ci_97.5']:.4f}]" if pd.notna(row["ci_2.5"]) and pd.notna(row["ci_97.5"]) else "",
        axis=1,
    )
    return df[["model", "metric", "bootstrap_mean", "95% CI"]].rename(
        columns={"model": "Model", "metric": "Metric", "bootstrap_mean": "Bootstrap mean"}
    )


def build_table_4_shap_topn(top_n: int = 10) -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage4_catboost_shap_importance.csv")
    if df.empty:
        return df
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).copy()
    df["Rank"] = range(1, len(df) + 1)
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce").round(6)
    return df[["Rank", "feature", "mean_abs_shap"]].rename(columns={"feature": "Feature", "mean_abs_shap": "Mean |SHAP|"})


def build_table_4b_category_shap(feature_name: str, top_n: int = 10) -> pd.DataFrame:
    path = OUTPUT_DIR / f"stage4_{feature_name}_shap_details.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return df
    for column in ["n", "mean_shap", "median_shap", "mean_abs_shap"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values(["mean_shap", "mean_abs_shap", "n"], ascending=[False, False, False]).head(top_n)
    feature_label = "Occupation major code" if feature_name == "occupation_major" else "Industry major code"
    return df.rename(
        columns={
            "category": feature_label,
            "n": "N",
            "mean_shap": "Mean SHAP",
            "median_shap": "Median SHAP",
            "mean_abs_shap": "Mean absolute SHAP",
            "direction": "Direction",
        }
    )


def build_table_5_segment_performance() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage4_segment_performance.csv")
    if df.empty:
        return df
    metric_cols = ["event_rate", "roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    df = df[df["model"].isin(["catboost", "hybrid_stack"])].copy()
    df = df[df["group_col"].isin(["age_group", "firm_size_group"])].copy()
    return df.rename(
        columns={
            "model": "Model",
            "group_col": "Grouping variable",
            "group_value": "Group",
            "n": "N",
            "event_rate": "Event rate",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "f1": "F1",
            "brier": "Brier score",
            "recall_at_10": "Recall@10",
            "lift_at_10": "Lift@10",
            "recall_at_20": "Recall@20",
            "lift_at_20": "Lift@20",
        }
    )


def build_table_6_robustness() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage4_robustness_2plus_firm.csv")
    if df.empty:
        return df
    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    return df.rename(
        columns={
            "model": "Model",
            "n": "N",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "f1": "F1",
            "brier": "Brier score",
            "recall_at_10": "Recall@10",
            "lift_at_10": "Lift@10",
            "recall_at_20": "Recall@20",
            "lift_at_20": "Lift@20",
        }
    )


def build_table_7_meta_coefficients() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_hybrid_meta_coefficients.csv")
    if df.empty:
        return df
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce").round(6)
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    return df.rename(columns={"meta_feature": "Meta-feature", "coefficient": "Coefficient"})


def build_appendix_wide_performance() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_hybrid_metrics.csv")
    if df.empty:
        return df
    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    wide = df.pivot(index="model", columns="split", values=metric_cols)
    wide.columns = [f"{metric}_{split}" for metric, split in wide.columns]
    return wide.reset_index().rename(columns={"model": "Model"})


def build_appendix_person_overlap() -> pd.DataFrame:
    return read_csv_if_exists(PROJECT_DIR / "sr_additional_analyses" / "person_level_sensitivity" / "person_overlap_audit.csv")


def build_appendix_person_disjoint() -> pd.DataFrame:
    return read_csv_if_exists(PROJECT_DIR / "sr_additional_analyses" / "person_level_sensitivity" / "person_disjoint_test_metrics.csv")


def build_appendix_cluster_bootstrap() -> pd.DataFrame:
    return read_csv_if_exists(PROJECT_DIR / "sr_additional_analyses" / "person_level_sensitivity" / "cluster_bootstrap_ci.csv")


def build_appendix_threshold_summary() -> pd.DataFrame:
    return read_csv_if_exists(PROJECT_DIR / "sr_additional_analyses" / "posthoc_recalibration" / "recalibration_threshold_diagnostics.csv")


def build_appendix_optional_deep_tabular() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "tables" / "TableS32_optional_deep_tabular_baseline.csv")
    if df.empty:
        return df
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier",
        "recall_at_20",
        "lift_at_20",
        "f1_at_valid_threshold",
        "valid_selected_threshold",
        "train_seconds",
        "test_inference_seconds",
    ]
    return format_metric_cols(df, metric_cols)


def build_appendix_feature_block_ablation() -> pd.DataFrame:
    return read_csv_if_exists(OUTPUT_DIR / "tables" / "TableS33_feature_block_incremental_ablation.csv")


def build_appendix_hours_removal_sensitivity() -> pd.DataFrame:
    return read_csv_if_exists(OUTPUT_DIR / "tables" / "TableS34_hours_block_removal_sensitivity.csv")


def build_appendix_destination_stratified() -> pd.DataFrame:
    return read_csv_if_exists(OUTPUT_DIR / "tables" / "TableS35_destination_stratified_diagnostics.csv")


def main() -> None:
    logger.info("==== Stage 5 paper tables start ====")
    save_table(build_table_1_dataset_summary(), "table_1_dataset_summary")
    save_table(build_table_1b_feature_space_summary(), "table_1b_feature_space_summary")
    save_table(build_table_2_main_model_performance(), "table_2_main_model_performance")
    save_table(build_table_3_bootstrap_ci(), "table_3_bootstrap_ci")
    save_table(build_table_4_shap_topn(top_n=10), "table_4_shap_top10")
    save_table(build_table_4b_category_shap("occupation_major", top_n=10), "table_4b_occupation_shap_details")
    save_table(build_table_4b_category_shap("industry_major", top_n=10), "table_4c_industry_shap_details")
    save_table(build_table_5_segment_performance(), "table_5_segment_performance")
    save_table(build_table_6_robustness(), "table_6_robustness_2plus_firm")
    save_table(build_table_7_meta_coefficients(), "table_7_hybrid_meta_coefficients")
    save_table(build_appendix_wide_performance(), "appendix_wide_model_performance")
    save_table(build_appendix_person_overlap(), "appendix_person_overlap_audit")
    save_table(build_appendix_person_disjoint(), "appendix_person_disjoint_test_metrics")
    save_table(build_appendix_cluster_bootstrap(), "appendix_cluster_bootstrap_ci")
    save_table(build_appendix_threshold_summary(), "appendix_threshold_diagnostics")
    save_table(build_appendix_optional_deep_tabular(), "appendix_optional_deep_tabular_baseline")
    save_table(build_appendix_feature_block_ablation(), "appendix_feature_block_incremental_ablation")
    save_table(build_appendix_hours_removal_sensitivity(), "appendix_hours_block_removal_sensitivity")
    save_table(build_appendix_destination_stratified(), "appendix_destination_stratified_diagnostics")

    summary = {
        "tables_created": [
            "table_1_dataset_summary",
            "table_1b_feature_space_summary",
            "table_2_main_model_performance",
            "table_3_bootstrap_ci",
            "table_4_shap_top10",
            "table_4b_occupation_shap_details",
            "table_4c_industry_shap_details",
            "table_5_segment_performance",
            "table_6_robustness_2plus_firm",
            "table_7_hybrid_meta_coefficients",
            "appendix_wide_model_performance",
            "appendix_person_overlap_audit",
            "appendix_person_disjoint_test_metrics",
            "appendix_cluster_bootstrap_ci",
            "appendix_threshold_diagnostics",
            "appendix_optional_deep_tabular_baseline",
            "appendix_feature_block_incremental_ablation",
            "appendix_hours_block_removal_sensitivity",
            "appendix_destination_stratified_diagnostics",
        ]
    }
    with open(TABLE_DIR / "stage5_tables_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logger.info("==== Stage 5 paper tables end ====")


if __name__ == "__main__":
    main()
