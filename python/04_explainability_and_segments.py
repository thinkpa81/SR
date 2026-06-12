
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_lib.modeling import (
    evaluate_binary_classifier,
    select_model_columns,
    setup_logging,
    split_chronological as timewise_split,
)
from pipeline_lib.paths import resolve_project_dir

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:                                                                
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:                                                           
    HAS_SHAP = False


PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"

                                                                  
                                                                     
TRAIN_END_WAVE = 20
VALID_END_WAVE = 23

TARGET_COL = "exit_label_t1"
ID_COLS = ("pid", "wave")
RANDOM_STATE = 42

                                                                            
MISSING_CATEGORY = "__MISSING__"
                                                                               
SHAP_SAMPLE_N = 5000
                                                                         
MIN_SUBGROUP_N = 100

                                                
AGE_BINS = [15, 24, 54, 64, np.inf]
AGE_LABELS = ["15-24", "25-54", "55-64", "65+"]
WEEKLY_HOURS_BINS = [0, 19, 29, 34, 39, np.inf]
WEEKLY_HOURS_LABELS = ["1-19", "20-29", "30-34", "35-39", "40+"]

setup_logging(LOG_DIR / "klips_stage4_explainability_segment.log")
logger = logging.getLogger(__name__)


def prepare_catboost_input(X: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    X = X.copy().replace({pd.NA: np.nan})

    for column in numeric_features:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in categorical_features:
        if column in X.columns:
            X[column] = X[column].astype(object)
            X.loc[pd.isna(X[column]), column] = MISSING_CATEGORY
            X[column] = X[column].astype(str)

    return X


def add_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "age_final" in out.columns:
        out["age_group"] = pd.cut(
            out["age_final"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=True,
            include_lowest=True,
        )

    if "weekly_hours" in out.columns:
        out["weekly_hours_group"] = pd.cut(
            out["weekly_hours"],
            bins=WEEKLY_HOURS_BINS,
            labels=WEEKLY_HOURS_LABELS,
            right=True,
        )

    if "one_person_firm_flag" in out.columns:
        out["firm_size_group"] = np.where(out["one_person_firm_flag"] == 1, "1-person", "2+-person")

    return out


def evaluate_by_group(df: pd.DataFrame, prob_col: str, group_col: str, model_name: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for group_value, subgroup in df.groupby(group_col, dropna=False):
        if len(subgroup) < MIN_SUBGROUP_N:
            continue
        y_true = subgroup[TARGET_COL].to_numpy()
        y_prob = subgroup[prob_col].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue

        row = {
            "model": model_name,
            "group_col": group_col,
            "group_value": str(group_value),
            "n": len(subgroup),
            "event_rate": float(np.mean(y_true)),
        }
        row.update(evaluate_binary_classifier(y_true, y_prob))
        rows.append(row)

    return pd.DataFrame(rows)


def save_categorical_shap_details(
    X_shap: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    target_features: list[str],
    output_dir: Path,
) -> list[str]:
    saved_files: list[str] = []
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    for feature in target_features:
        if feature not in X_shap.columns or feature not in shap_df.columns:
            continue

        detail_df = pd.DataFrame(
            {
                "category": X_shap[feature].astype(str).values,
                "shap_value": pd.to_numeric(shap_df[feature], errors="coerce"),
            }
        )
        detail_df = detail_df.dropna(subset=["shap_value"])
        if detail_df.empty:
            continue

        summary_df = (
            detail_df.groupby("category", dropna=False)
            .agg(
                n=("shap_value", "size"),
                mean_shap=("shap_value", "mean"),
                median_shap=("shap_value", "median"),
                mean_abs_shap=("shap_value", lambda series: np.abs(series).mean()),
            )
            .reset_index()
        )
        summary_df["direction"] = np.where(summary_df["mean_shap"] > 0, "higher_exit_risk", "lower_exit_risk")
        summary_df = summary_df.sort_values(["mean_shap", "mean_abs_shap", "n"], ascending=[False, False, False])

        csv_path = output_dir / f"stage4_{feature}_shap_details.csv"
        xlsx_path = output_dir / f"stage4_{feature}_shap_details.xlsx"
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Sheet1")

        saved_files.extend([csv_path.name, xlsx_path.name])
        logger.info("Saved categorical SHAP detail file: %s", csv_path.name)

    return saved_files


def main() -> None:
    logger.info("==== Stage 4 explainability and subgroup evaluation start ====")

    analysis_path = PROCESSED_DIR / "analysis_base_with_label.csv"
    prediction_path = OUTPUT_DIR / "stage3_test_predictions_with_hybrid.csv"

    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing file: {analysis_path}")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing file: {prediction_path}")

    df = pd.read_csv(analysis_path)
    pred_df = pd.read_csv(prediction_path)

    logger.info("Analysis base shape=%s", df.shape)
    logger.info("Prediction file shape=%s", pred_df.shape)

    train_df, valid_df, test_df = timewise_split(df, train_end=TRAIN_END_WAVE, valid_end=VALID_END_WAVE)
    logger.info("Train=%s, Valid=%s, Test=%s", train_df.shape, valid_df.shape, test_df.shape)

    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    shap_detail_files: list[str] = []

    if not HAS_CATBOOST:
        logger.warning("catboost is not installed. CatBoost SHAP and robustness steps are skipped.")
    else:
        X_train = prepare_catboost_input(train_df[feature_cols], numeric_features, categorical_features)
        y_train = train_df[TARGET_COL].astype(int)

        X_test = prepare_catboost_input(test_df[feature_cols], numeric_features, categorical_features)
        y_test = test_df[TARGET_COL].astype(int)

        cat_feature_index = [X_train.columns.get_loc(column) for column in categorical_features if column in X_train.columns]

                                                                            
        cb_model = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=RANDOM_STATE,
        )
        cb_model.fit(X_train, y_train, cat_features=cat_feature_index)

        cb_test_prob = cb_model.predict_proba(X_test)[:, 1]
        cb_metrics = evaluate_binary_classifier(y_test.to_numpy(), cb_test_prob)
        pd.DataFrame([{"model": "catboost_retrain_test", **cb_metrics}]).to_csv(
            OUTPUT_DIR / "stage4_catboost_retrain_test_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if HAS_SHAP:
            shap_sample_n = min(SHAP_SAMPLE_N, len(X_test))
            X_shap = X_test.sample(n=shap_sample_n, random_state=RANDOM_STATE).copy()

            explainer = shap.TreeExplainer(cb_model)
            shap_values = explainer.shap_values(X_shap)
                                                                                        
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs_shap}).sort_values(
                "mean_abs_shap", ascending=False
            )
            shap_importance.to_csv(
                OUTPUT_DIR / "stage4_catboost_shap_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )

            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_shap, show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "stage4_catboost_shap_summary.png", dpi=200, bbox_inches="tight")
            plt.close()

            shap_detail_files = save_categorical_shap_details(
                X_shap=X_shap,
                shap_values=shap_values,
                feature_names=X_shap.columns.tolist(),
                target_features=["occupation_major", "industry_major"],
                output_dir=OUTPUT_DIR,
            )
            logger.info("SHAP outputs saved.")
        else:
            logger.warning("shap is not installed. SHAP analysis is skipped.")

                                                                                      
        if "one_person_firm_flag" in test_df.columns:
            test_two_plus = test_df[test_df["one_person_firm_flag"] != 1].copy()
            if len(test_two_plus) > 0:
                X_test_two_plus = prepare_catboost_input(test_two_plus[feature_cols], numeric_features, categorical_features)
                y_test_two_plus = test_two_plus[TARGET_COL].astype(int).to_numpy()
                prob_two_plus = cb_model.predict_proba(X_test_two_plus)[:, 1]

                robustness_metrics = evaluate_binary_classifier(y_test_two_plus, prob_two_plus)
                pd.DataFrame([{"model": "catboost_2plus_firm_test", "n": len(test_two_plus), **robustness_metrics}]).to_csv(
                    OUTPUT_DIR / "stage4_robustness_2plus_firm.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                                                                                      
    merge_cols = [column for column in (*ID_COLS, TARGET_COL) if column in pred_df.columns and column in test_df.columns]
    test_eval = test_df.merge(pred_df, on=merge_cols, how="left")
    test_eval = add_segment_columns(test_eval)

    prob_cols = [column for column in test_eval.columns if column.startswith("proba_")]
    segment_results = []
    for prob_col in prob_cols:
        model_name = prob_col.replace("proba_", "")
        for group_col in ["age_group", "weekly_hours_group", "firm_size_group"]:
            segment_df = evaluate_by_group(test_eval, prob_col, group_col, model_name)
            if not segment_df.empty:
                segment_results.append(segment_df)

    if segment_results:
        segment_out = pd.concat(segment_results, ignore_index=True)
        segment_out.to_csv(OUTPUT_DIR / "stage4_segment_performance.csv", index=False, encoding="utf-8-sig")
        logger.info("Segment performance saved.")

    summary = {
        "analysis_shape": list(df.shape),
        "test_shape": list(test_df.shape),
        "prediction_columns": prob_cols,
        "shap_enabled": HAS_SHAP,
        "catboost_enabled": HAS_CATBOOST,
        "categorical_shap_detail_files": shap_detail_files,
    }

    with open(OUTPUT_DIR / "stage4_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logger.info("==== Stage 4 explainability and subgroup evaluation end ====")


if __name__ == "__main__":
    main()
