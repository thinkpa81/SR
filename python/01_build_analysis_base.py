
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipeline_lib.features import build_feature_manifest
from pipeline_lib.modeling import (
    evaluate_binary_classifier,
    fit_sklearn_pipeline,
    sanitize_model_input,
    select_model_columns,
    setup_logging,
    split_chronological as timewise_split,
)
from pipeline_lib.paths import resolve_project_dir

PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

RAW_DIR_CANDIDATES = [
    PROJECT_DIR / "raw",
    PROJECT_DIR,
]


def resolve_raw_dir(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


RAW_DIR = resolve_raw_dir(RAW_DIR_CANDIDATES)
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
INTERIM_DIR = OUTPUT_DIR / "interim"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"
TEMP_DIR = OUTPUT_DIR / "temp"
DEFAULT_XLSX_FIX_DIR = Path(tempfile.gettempdir()) / "klips_xlsx_fix"
XLSX_FIX_DIR = Path(os.environ.get("KLIPS_XLSX_FIX_DIR", str(DEFAULT_XLSX_FIX_DIR)))

for directory in [PROJECT_DIR, OUTPUT_DIR, INTERIM_DIR, PROCESSED_DIR, LOG_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

setup_logging(LOG_DIR / "klips_sr_pipeline.log")
logger = logging.getLogger(__name__)


                                                                               
                                                                        
SPECIAL_MISSING_VALUES = {
    -1, -2, -3, -4, -7, -8, -9,
    999, 9999, 99999, 999999, 9999999,
}

                                                                             
                                                                            
                                             
WAGE_WORKER_STATUS_CODES = [1, 2, 3]
NON_WAGE_WORKER_STATUS_CODES = [4, 5, 6]


@dataclass
class FileMeta:
    path: Path
    wave: int
    source_type: str
    filename: str


@dataclass
class ConceptRule:
    concept: str
    source_type: str
    pattern: str
    description: str = ""


def discover_klips_files(raw_dir: Path) -> list[FileMeta]:
    metas: list[FileMeta] = []
    regex = re.compile(r"klips(\d{2})([ahpw])(?:\d)?\.(xlsx|xls)$", re.IGNORECASE)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Configured RAW_DIR does not exist: {raw_dir}")

    all_files = [path for path in raw_dir.rglob("*") if path.is_file()]
    logger.info("Search root: %s", raw_dir)
    logger.info("Total files found under root: %s", len(all_files))

    for path in sorted(all_files):
        match = regex.search(path.name)
        if not match:
            continue
        wave = int(match.group(1))
        source_type = match.group(2).lower()
        metas.append(FileMeta(path=path, wave=wave, source_type=source_type, filename=path.name))

    logger.info("Discovered %s KLIPS files", len(metas))
    if metas:
        logger.info("Sample discovered files: %s", [meta.filename for meta in metas[:10]])
    return metas


def build_inventory(file_metas: list[FileMeta]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meta in file_metas:
        try:
            preview_df = read_excel_safely(meta.path, nrows=0) if meta.source_type in {"p", "h"} else pd.DataFrame()
            rows.append(
                {
                    "filename": meta.filename,
                    "full_path": str(meta.path),
                    "wave": meta.wave,
                    "source_type": meta.source_type,
                    "n_preview_rows": len(preview_df),
                    "n_preview_cols": preview_df.shape[1],
                    "columns_preview": ", ".join(map(str, preview_df.columns[:10]))
                    if meta.source_type in {"p", "h"} else "header scan skipped for non-core supplemental/work-history file",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "filename": meta.filename,
                    "full_path": str(meta.path),
                    "wave": meta.wave,
                    "source_type": meta.source_type,
                    "n_preview_rows": np.nan,
                    "n_preview_cols": np.nan,
                    "columns_preview": f"READ_ERROR: {exc}",
                }
            )

    inventory = pd.DataFrame(rows).sort_values(["wave", "source_type", "filename"])
    inventory.to_csv(OUTPUT_DIR / "inventory_preview.csv", index=False, encoding="utf-8-sig")
    return inventory


def sanitize_xlsx_synchvertical(src_path: Path) -> Path:
    XLSX_FIX_DIR.mkdir(parents=True, exist_ok=True)
    temp_subdir = XLSX_FIX_DIR / f"{src_path.stem}_{uuid.uuid4().hex}"
    temp_subdir.mkdir(parents=True, exist_ok=True)
    fixed_path = temp_subdir / src_path.name

    with ZipFile(src_path, "r") as zip_in, ZipFile(
        fixed_path,
        "w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as zip_out:
        for item in zip_in.infolist():
            data = zip_in.read(item.filename)
            if item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
                data = re.sub(br'\s+synchVertical="[^"]*"', b"", data)
            zip_out.writestr(item, data)

    if not fixed_path.exists() or fixed_path.stat().st_size == 0:
        raise FileNotFoundError(f"Sanitized workbook was not created: {fixed_path}")
    return fixed_path


def read_excel_safely(
    path: Path,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported file extension: {path}")

    if suffix == ".xls":
        try:
            df = pd.read_excel(path, dtype=object, nrows=nrows, usecols=usecols)
        except Exception:
            df = pd.read_excel(path, nrows=nrows, usecols=usecols)
        df.columns = [str(column).strip() for column in df.columns]
        return df

    try:
        df = pd.read_excel(path, dtype=object, nrows=nrows, usecols=usecols, engine="calamine")
        df.columns = [str(column).strip() for column in df.columns]
        return df
    except Exception as exc:
        logger.warning("calamine read failed for %s; falling back to openpyxl/sanitized read: %s", path, exc)

    try:
        df = pd.read_excel(path, dtype=object, nrows=nrows, usecols=usecols)
        df.columns = [str(column).strip() for column in df.columns]
        return df
    except TypeError as exc:
        if "synchVertical" not in str(exc):
            raise

        logger.warning("openpyxl synchVertical issue detected: %s", path)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            fixed_path = sanitize_xlsx_synchvertical(path)
            logger.warning("Retry with sanitized workbook (%s/3): %s", attempt, fixed_path)
            try:
                df = pd.read_excel(fixed_path, dtype=object, nrows=nrows, usecols=usecols)
                break
            except Exception as retry_exc:
                last_error = retry_exc
                try:
                    df = pd.read_excel(fixed_path, nrows=nrows, usecols=usecols)
                    break
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    logger.warning(
                        "Sanitized workbook read failed (%s/3), regenerating if possible: %s",
                        attempt,
                        fallback_exc,
                    )
        else:
            raise RuntimeError(f"Could not read sanitized workbook for {path}") from last_error

        df.columns = [str(column).strip() for column in df.columns]
        return df


def add_wave_and_year(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    out = df.copy()
    out["wave"] = wave
    out["survey_year"] = 1997 + wave
    return out


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({pd.NA: np.nan})
    for column in out.columns:
        try:
            out[column] = out[column].replace(list(SPECIAL_MISSING_VALUES), np.nan)
        except (TypeError, ValueError) as exc:
            logger.debug("Numeric missing-value replacement skipped for %s: %s", column, exc)
        if out[column].dtype == "object":
            out[column] = out[column].replace({str(value): np.nan for value in SPECIAL_MISSING_VALUES})
    return out


CONCEPT_RULES: list[ConceptRule] = [
    ConceptRule("pid", "p", r"^pid$", "person identifier"),
    ConceptRule("gender_raw", "p", r"p\d{2}0101$", "gender"),
    ConceptRule("relation_to_head_raw", "p", r"p\d{2}0102$", "relation to household head"),
    ConceptRule("birth_year_raw", "p", r"p\d{2}0104$", "birth year"),
    ConceptRule("age_reported_raw", "p", r"p\d{2}0107$", "reported age"),
    ConceptRule("region_raw", "p", r"p\d{2}0121$", "residence region"),
    ConceptRule("education_level_raw", "p", r"p\d{2}0110$", "education level"),
    ConceptRule("education_completion_raw", "p", r"p\d{2}0111$", "schooling completion"),
    ConceptRule("education_grade_raw", "p", r"p\d{2}0112$", "education grade"),
    ConceptRule("major_field_raw", "p", r"p\d{2}0113$", "major field"),
    ConceptRule("marital_status_raw", "p", r"p\d{2}5501$", "marital status"),
    ConceptRule("health_status_raw", "p", r"p\d{2}6101$", "self-rated health"),
    ConceptRule("health_change_raw", "p", r"p\d{2}6102$", "health change relative to one year earlier"),

    ConceptRule("employment_status_raw", "p", r"p\d{2}0201$", "economic activity status"),
    ConceptRule("employment_type_raw", "p", r"p\d{2}0211$", "employment type"),
    ConceptRule("family_work_help_raw", "p", r"p\d{2}0212$", "unpaid family work"),
    ConceptRule("employee_status_raw", "p", r"p\d{2}0314$", "employment status category"),
    ConceptRule("work_type_raw", "p", r"p\d{2}0315$", "work type"),
    ConceptRule("job_position_raw", "p", r"p\d{2}0316$", "job position"),
    ConceptRule("regular_worker_raw", "p", r"p\d{2}0317$", "regular worker indicator"),
    ConceptRule("business_place_raw", "p", r"p\d{2}0318$", "description of place of business"),
    ConceptRule("workplace_registered_raw", "p", r"p\d{2}0319$", "business registration"),

    ConceptRule("job_start_year_raw", "p", r"p\d{2}0301$", "current job start year"),
    ConceptRule("job_start_month_raw", "p", r"p\d{2}0302$", "current job start month"),
    ConceptRule("job_start_day_raw", "p", r"p\d{2}0303$", "current job start day"),
    ConceptRule("industry_raw", "p", r"p\d{2}034[012]$", "industry code across KSIC revisions"),
    ConceptRule("occupation_raw", "p", r"p\d{2}035[012]$", "occupation code across KSCO revisions"),
    ConceptRule("workplace_type_raw", "p", r"p\d{2}0401$", "workplace type"),
    ConceptRule("firm_size_raw", "p", r"p\d{2}0402$", "firm size"),
    ConceptRule("firm_size_band_raw", "p", r"p\d{2}0403$", "firm-size category"),
    ConceptRule("worksite_size_raw", "p", r"p\d{2}0405$", "worksite size"),
    ConceptRule("worksite_size_band_raw", "p", r"p\d{2}0406$", "worksite-size category"),
    ConceptRule("contract_period_raw", "p", r"p\d{2}0501$", "fixed contract period"),
    ConceptRule("written_contract_raw", "p", r"p\d{2}0509$", "written work contract"),
    ConceptRule("expected_duration_raw", "p", r"p\d{2}0603$", "expected continued employment duration"),
    ConceptRule("voluntary_work_form_raw", "p", r"p\d{2}0614$", "voluntary or involuntary work form"),
    ConceptRule("fixed_work_hours_raw", "p", r"p\d{2}1003$", "fixed work hours"),
    ConceptRule("weekly_hours_raw", "p", r"p\d{2}1004$", "average weekly working hours"),
    ConceptRule("weekly_work_days_raw", "p", r"p\d{2}1005$", "average weekly working days"),
    ConceptRule("regular_weekly_hours_raw", "p", r"p\d{2}1006$", "regular weekly work hours"),
    ConceptRule("regular_weekly_days_raw", "p", r"p\d{2}1007$", "regular weekly work days"),
    ConceptRule("overtime_work_raw", "p", r"p\d{2}1011$", "overtime work"),
    ConceptRule("overtime_hours_raw", "p", r"p\d{2}1012$", "reported overtime hours"),
    ConceptRule("overtime_days_raw", "p", r"p\d{2}1013$", "reported overtime days"),
    ConceptRule("overtime_pay_raw", "p", r"p\d{2}1018$", "monthly overtime payment"),
    ConceptRule("pay_schedule_raw", "p", r"p\d{2}1601$", "wage pay schedule"),
    ConceptRule("pay_method_raw", "p", r"p\d{2}1602$", "wage determination method"),
    ConceptRule("performance_pay_raw", "p", r"p\d{2}1621$", "performance-based pay"),
    ConceptRule("monthly_wage_raw", "p", r"p\d{2}1642$", "average monthly wage"),
    ConceptRule("monthly_tax_deduction_raw", "p", r"p\d{2}1643$", "monthly tax deduction"),
    ConceptRule("starting_monthly_wage_raw", "p", r"p\d{2}1652$", "monthly wage when starting job"),
    ConceptRule("annual_earned_income_raw", "p", r"p\d{2}1702$", "annual earned income"),
    ConceptRule("aftertax_earned_income_raw", "p", r"p\d{2}1703$", "after-tax annual earned income"),
    ConceptRule("national_pension_raw", "p", r"p\d{2}2101$", "national pension coverage"),
    ConceptRule("health_insurance_raw", "p", r"p\d{2}2103$", "workplace health insurance coverage"),
    ConceptRule("employment_insurance_raw", "p", r"p\d{2}2104$", "employment insurance"),
    ConceptRule("industrial_accident_insurance_raw", "p", r"p\d{2}2105$", "industrial accident compensation insurance"),
    ConceptRule("current_employment_insurance_raw", "p", r"p\d{2}2109$", "current employment insurance coverage"),
    ConceptRule("union_exists_raw", "p", r"p\d{2}2501$", "union exists at main job"),
    ConceptRule("union_member_raw", "p", r"p\d{2}2504$", "union membership"),
    ConceptRule("shift_work_raw", "p", r"p\d{2}2601$", "shift work"),
    ConceptRule("job_sat_wage_raw", "p", r"p\d{2}4311$", "satisfaction with wage or earnings"),
    ConceptRule("job_sat_stability_raw", "p", r"p\d{2}4312$", "satisfaction with employment stability"),
    ConceptRule("job_sat_content_raw", "p", r"p\d{2}4313$", "satisfaction with work content"),
    ConceptRule("job_sat_environment_raw", "p", r"p\d{2}4314$", "satisfaction with work environment"),
    ConceptRule("job_sat_hours_raw", "p", r"p\d{2}4315$", "satisfaction with work hours"),
    ConceptRule("job_sat_development_raw", "p", r"p\d{2}4316$", "satisfaction with personal development"),
    ConceptRule("job_sat_relationships_raw", "p", r"p\d{2}4317$", "satisfaction with workplace relationships"),
    ConceptRule("job_sat_promotion_raw", "p", r"p\d{2}4318$", "satisfaction with promotion fairness"),
    ConceptRule("job_sat_welfare_raw", "p", r"p\d{2}4319$", "satisfaction with corporate welfare"),
    ConceptRule("job_sat_overall_workplace_raw", "p", r"p\d{2}4321$", "overall satisfaction with workplace"),
    ConceptRule("job_sat_overall_work_raw", "p", r"p\d{2}4322$", "overall satisfaction with work itself"),

    ConceptRule("household_size_raw", "h", r"h\d{2}0150$", "household size"),
    ConceptRule("housing_tenure_raw", "h", r"h\d{2}1406$", "housing tenure"),
    ConceptRule("housing_type_raw", "h", r"h\d{2}1407$", "housing type"),
    ConceptRule("housing_deposit_raw", "h", r"h\d{2}1413$", "housing deposit"),
    ConceptRule("housing_rent_raw", "h", r"h\d{2}1414$", "monthly rent"),
    ConceptRule("household_labor_income_raw", "h", r"h\d{2}2102$", "previous-year household earned income"),
    ConceptRule("household_monthly_earned_income_raw", "h", r"h\d{2}2202$", "previous-month household earned income"),
    ConceptRule("household_medical_cost_raw", "h", r"h\d{2}2318$", "previous-year household medical cost"),
    ConceptRule("household_monthly_savings_raw", "h", r"h\d{2}2402$", "average monthly household savings"),
    ConceptRule("household_bank_savings_raw", "h", r"h\d{2}2562$", "bank savings amount"),
    ConceptRule("household_debt_exists_raw", "h", r"h\d{2}2632$", "household debt exists"),
    ConceptRule("household_financial_condition_raw", "h", r"h\d{2}2705$", "current household financial condition"),
]


def find_columns_by_rule(columns: Iterable[str], rule: ConceptRule) -> list[str]:
    pattern = re.compile(rule.pattern, re.IGNORECASE)
    return [column for column in columns if pattern.search(str(column))]


def extract_concepts(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for rule in CONCEPT_RULES:
        if rule.source_type != source_type:
            continue
        matched = find_columns_by_rule(df.columns, rule)
        if not matched:
            continue
        if len(matched) > 1:
            logger.warning("Multiple columns matched for %s: %s", rule.concept, matched)
                                                                                
                                                                             
                                    
        out[rule.concept] = df[matched].bfill(axis=1).iloc[:, 0]

    if "pid" not in out.columns:
        for candidate in ["pid", "PID", "person_id", "PERSON_ID"]:
            if candidate in df.columns:
                out["pid"] = df[candidate]
                break

    if "hhid" not in out.columns:
        wave_value = None
        if "wave" in df.columns and not df["wave"].dropna().empty:
            wave_value = int(df["wave"].dropna().iloc[0])
        wave_candidates = []
        if wave_value is not None:
            wave_candidates = [f"hhid{wave_value:02d}", f"HHID{wave_value:02d}"]
        for candidate in [*wave_candidates, "hhid", "HHID", "household_id", "HOUSEHOLD_ID"]:
            if candidate in df.columns:
                out["hhid"] = df[candidate]
                break

    return out


def required_columns_for_meta(meta: FileMeta) -> list[str]:
    header_df = read_excel_safely(meta.path, nrows=0)
    columns = [str(column).strip() for column in header_df.columns]
    required: list[str] = []

    for rule in CONCEPT_RULES:
        if rule.source_type != meta.source_type:
            continue
        required.extend(find_columns_by_rule(columns, rule))

    required.extend(["pid", "PID", f"hhid{meta.wave:02d}", f"HHID{meta.wave:02d}", "hhid", "HHID"])
    selected = [column for column in dict.fromkeys(required) if column in columns]
    if not selected:
        logger.warning("No selected columns resolved for %s; falling back to full read.", meta.filename)
        return columns
    return selected


def load_source_panels(file_metas: list[FileMeta]) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_frames: list[pd.DataFrame] = []
    household_frames: list[pd.DataFrame] = []

    for meta in file_metas:
        if meta.source_type not in {"p", "h"}:
            continue

        logger.info("Reading %s", meta.path)
        usecols = required_columns_for_meta(meta)
        logger.info("Selected %s/%s columns for %s", len(usecols), "?", meta.filename)
        raw = read_excel_safely(meta.path, usecols=usecols)
        raw = add_wave_and_year(raw, meta.wave)
        raw = normalize_missing_values(raw)
        extracted = extract_concepts(raw, meta.source_type)
        extracted["wave"] = meta.wave
        extracted["survey_year"] = 1997 + meta.wave
        extracted["source_file"] = meta.filename

        if meta.source_type == "p":
            person_frames.append(extracted)
        elif meta.source_type == "h":
            household_frames.append(extracted)

    person_df = pd.concat(person_frames, ignore_index=True, sort=False) if person_frames else pd.DataFrame()
    household_df = pd.concat(household_frames, ignore_index=True, sort=False) if household_frames else pd.DataFrame()

    return person_df, household_df


def build_panel_master(person_df: pd.DataFrame, household_df: pd.DataFrame) -> pd.DataFrame:
    if person_df.empty:
        raise ValueError("person_df is empty")

    panel = person_df.copy()
    if not household_df.empty and "hhid" in panel.columns and "hhid" in household_df.columns:
        household_cols = [column for column in household_df.columns if column not in {"source_file"}]
        household_cols = list(dict.fromkeys(household_cols))
        household_use = household_df[household_cols].copy()

        merge_keys = [column for column in ["hhid", "wave", "survey_year"] if column in household_use.columns and column in panel.columns]
        household_non_keys = [column for column in household_use.columns if column not in merge_keys]
        household_non_keys = [column for column in household_non_keys if column not in panel.columns]

        panel = panel.merge(household_use[merge_keys + household_non_keys], on=merge_keys, how="left")

    subset_keys = [column for column in ["pid", "wave"] if column in panel.columns]
    if subset_keys:
        panel = panel.drop_duplicates(subset=subset_keys)

    panel.to_csv(INTERIM_DIR / "panel_master_raw.csv", index=False, encoding="utf-8-sig")
    logger.info("Panel master shape: %s", panel.shape)
    return panel


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_age_from_birth_year(birth_year: pd.Series, survey_year: pd.Series) -> pd.Series:
    birth_year_num = to_numeric(birth_year)
    survey_year_num = to_numeric(survey_year)
    return survey_year_num - birth_year_num + 1


def map_gender(series: pd.Series) -> pd.Series:
    numeric_series = to_numeric(series)
    mapping = {1: "male", 2: "female"}
    return numeric_series.map(mapping)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def derive_employment_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for column in [
        "employment_status_raw",
        "employment_type_raw",
        "family_work_help_raw",
        "employee_status_raw",
        "regular_worker_raw",
        "weekly_hours_raw",
        "weekly_work_days_raw",
        "regular_weekly_hours_raw",
        "regular_weekly_days_raw",
        "overtime_hours_raw",
        "overtime_days_raw",
        "overtime_pay_raw",
        "monthly_wage_raw",
        "monthly_tax_deduction_raw",
        "starting_monthly_wage_raw",
        "annual_earned_income_raw",
        "aftertax_earned_income_raw",
        "job_start_year_raw",
        "job_start_month_raw",
        "birth_year_raw",
        "firm_size_raw",
        "worksite_size_raw",
        "household_size_raw",
        "housing_deposit_raw",
        "housing_rent_raw",
        "household_labor_income_raw",
        "household_monthly_earned_income_raw",
        "household_medical_cost_raw",
        "household_monthly_savings_raw",
        "household_bank_savings_raw",
        "household_debt_exists_raw",
        "job_sat_wage_raw",
        "job_sat_stability_raw",
        "job_sat_content_raw",
        "job_sat_environment_raw",
        "job_sat_hours_raw",
        "job_sat_development_raw",
        "job_sat_relationships_raw",
        "job_sat_promotion_raw",
        "job_sat_welfare_raw",
        "job_sat_overall_workplace_raw",
        "job_sat_overall_work_raw",
    ]:
        if column in out.columns:
            out[column] = to_numeric(out[column])

    if "gender_raw" in out.columns:
        out["gender"] = map_gender(out["gender_raw"])

    if "birth_year_raw" in out.columns:
        out["age_final"] = infer_age_from_birth_year(out["birth_year_raw"], out["survey_year"])

    if "employee_status_raw" in out.columns:
        out["is_wage_worker_t"] = out["employee_status_raw"].isin(WAGE_WORKER_STATUS_CODES).astype(float)
        out["is_non_wage_worker_t"] = out["employee_status_raw"].isin(NON_WAGE_WORKER_STATUS_CODES).astype(float)
    else:
        out["is_wage_worker_t"] = np.nan
        out["is_non_wage_worker_t"] = np.nan

    if "employment_status_raw" in out.columns:
        out["is_employed_t"] = out["employment_status_raw"].notna().astype(float)
    else:
        out["is_employed_t"] = np.nan

    return out


def add_panel_history_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"pid", "wave"}.issubset(df.columns):
        return df

    out = df.sort_values(["pid", "wave"]).copy()
    grouped = out.groupby("pid", sort=False)

    out["panel_observation_number"] = grouped.cumcount() + 1
    out["prior_observation_count"] = out["panel_observation_number"] - 1
    out["waves_since_first_observed"] = out["wave"] - grouped["wave"].transform("min")

    if "is_wage_worker_t" in out.columns:
        prior_wage = grouped["is_wage_worker_t"].shift(1)
        out["prior_wage_worker_count"] = prior_wage.groupby(out["pid"]).cumsum()
        out["prior_wage_worker_count"] = out["prior_wage_worker_count"].fillna(0)
        out["prior_non_wage_worker_count"] = out["prior_observation_count"] - out["prior_wage_worker_count"]
        out["prior_wage_worker_share"] = safe_divide(
            out["prior_wage_worker_count"],
            out["prior_observation_count"].replace(0, np.nan),
        )

    for source, target in [
        ("monthly_wage", "monthly_wage_lag1"),
        ("weekly_hours", "weekly_hours_lag1"),
        ("tenure_years", "tenure_years_lag1"),
        ("firm_size_raw", "firm_size_lag1"),
    ]:
        if source in out.columns:
            out[target] = grouped[source].shift(1)

    if {"monthly_wage", "monthly_wage_lag1"}.issubset(out.columns):
        out["wage_change_lag1"] = out["monthly_wage"] - out["monthly_wage_lag1"]
        out["wage_pct_change_lag1"] = safe_divide(out["wage_change_lag1"], out["monthly_wage_lag1"])
        out["wage_drop_flag"] = (out["wage_change_lag1"] < 0).astype(float)
        shifted = grouped["monthly_wage"].shift(1)
        out["wage_rolling3_mean"] = shifted.groupby(out["pid"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        out["wage_rolling3_std"] = shifted.groupby(out["pid"]).rolling(3, min_periods=2).std().reset_index(level=0, drop=True)

    if {"weekly_hours", "weekly_hours_lag1"}.issubset(out.columns):
        out["hours_change_lag1"] = out["weekly_hours"] - out["weekly_hours_lag1"]
        out["hours_pct_change_lag1"] = safe_divide(out["hours_change_lag1"], out["weekly_hours_lag1"])
        out["hours_drop_flag"] = (out["hours_change_lag1"] < 0).astype(float)
        shifted = grouped["weekly_hours"].shift(1)
        out["hours_rolling3_mean"] = shifted.groupby(out["pid"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        out["hours_rolling3_std"] = shifted.groupby(out["pid"]).rolling(3, min_periods=2).std().reset_index(level=0, drop=True)

    if {"tenure_years", "tenure_years_lag1"}.issubset(out.columns):
        out["tenure_change_lag1"] = out["tenure_years"] - out["tenure_years_lag1"]

    for source, target in [
        ("occupation_major", "occupation_changed_lag1"),
        ("industry_major", "industry_changed_lag1"),
        ("region", "region_changed_lag1"),
        ("firm_size_raw", "firm_size_changed_lag1"),
    ]:
        if source in out.columns:
            lag = grouped[source].shift(1)
            out[target] = ((out[source].notna()) & (lag.notna()) & (out[source].astype(str) != lag.astype(str))).astype(float)

    return out.sort_index()


def build_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = derive_employment_flags(df)

    if "age_final" in out.columns:
        out.loc[out["age_final"] < 15, "age_final"] = np.nan
        out["age_gt_90_flag"] = (out["age_final"] > 90).astype(float)
        out["age_squared"] = out["age_final"] ** 2
        out["young_worker_flag"] = (out["age_final"] < 30).astype(float)
        out["older_worker_flag"] = (out["age_final"] >= 55).astype(float)

    if {"job_start_year_raw", "job_start_month_raw", "survey_year"}.issubset(out.columns):
        survey_month_assumed = 12
        tenure_months = (out["survey_year"] - out["job_start_year_raw"]) * 12 + (
            survey_month_assumed - out["job_start_month_raw"]
        )
        out["tenure_months"] = tenure_months.where(tenure_months >= 0, np.nan)
        out["tenure_years"] = out["tenure_months"] / 12.0
        out["tenure_squared"] = out["tenure_years"] ** 2
        out["log_tenure_years"] = np.log1p(out["tenure_years"])
        out["short_tenure_flag"] = (out["tenure_years"] < 1).astype(float)
        out["long_tenure_flag"] = (out["tenure_years"] >= 10).astype(float)

    if "weekly_hours_raw" in out.columns:
        out["weekly_hours"] = out["weekly_hours_raw"].copy()
        out.loc[out["weekly_hours"] <= 0, "weekly_hours"] = np.nan
        out["weekly_hours_missing"] = out["weekly_hours"].isna().astype(float)
        out["gt48"] = (out["weekly_hours"] > 48).astype(float)
        out["gt52"] = (out["weekly_hours"] > 52).astype(float)
        out["ge55"] = (out["weekly_hours"] >= 55).astype(float)
        out["weekly_hours_qc_gt112"] = (out["weekly_hours"] > 112).astype(float)
        out["weekly_hours_band"] = pd.cut(
            out["weekly_hours"],
            bins=[0, 19, 29, 34, 39, np.inf],
            labels=["1-19", "20-29", "30-34", "35-39", "40+"],
            right=True,
        )

    for raw_name, feature_name in [
        ("weekly_work_days_raw", "weekly_work_days"),
        ("regular_weekly_hours_raw", "regular_weekly_hours"),
        ("regular_weekly_days_raw", "regular_weekly_days"),
        ("overtime_hours_raw", "overtime_hours"),
        ("overtime_days_raw", "overtime_days"),
        ("overtime_pay_raw", "overtime_pay"),
        ("monthly_tax_deduction_raw", "monthly_tax_deduction"),
        ("starting_monthly_wage_raw", "starting_monthly_wage"),
        ("annual_earned_income_raw", "annual_earned_income"),
        ("aftertax_earned_income_raw", "aftertax_earned_income"),
        ("worksite_size_raw", "worksite_size"),
        ("housing_deposit_raw", "housing_deposit"),
        ("housing_rent_raw", "housing_rent"),
        ("household_labor_income_raw", "household_annual_earned_income"),
        ("household_monthly_earned_income_raw", "household_monthly_earned_income"),
        ("household_medical_cost_raw", "household_medical_cost"),
        ("household_monthly_savings_raw", "household_monthly_savings"),
        ("household_bank_savings_raw", "household_bank_savings"),
    ]:
        if raw_name in out.columns:
            out[feature_name] = out[raw_name].copy()
            out.loc[out[feature_name] < 0, feature_name] = np.nan

    if "monthly_wage_raw" in out.columns:
        out["monthly_wage"] = out["monthly_wage_raw"].copy()
        out.loc[out["monthly_wage"] <= 0, "monthly_wage"] = np.nan
        out["monthly_wage_missing"] = out["monthly_wage"].isna().astype(float)
        out["log_monthly_wage"] = np.log1p(out["monthly_wage"])

    if "annual_earned_income" in out.columns:
        out["log_annual_earned_income"] = np.log1p(out["annual_earned_income"])
    if "aftertax_earned_income" in out.columns:
        out["log_aftertax_earned_income"] = np.log1p(out["aftertax_earned_income"])
    if "starting_monthly_wage" in out.columns:
        out["log_starting_monthly_wage"] = np.log1p(out["starting_monthly_wage"])

    if {"monthly_wage", "weekly_hours"}.issubset(out.columns):
        out["hourly_wage_proxy"] = safe_divide(out["monthly_wage"], out["weekly_hours"] * 4.345)
        out["log_hourly_wage_proxy"] = np.log1p(out["hourly_wage_proxy"])
        out["wage_x_hours"] = out["monthly_wage"] * out["weekly_hours"]
    if {"monthly_wage", "tenure_years"}.issubset(out.columns):
        out["wage_x_tenure"] = out["monthly_wage"] * out["tenure_years"]
    if {"weekly_hours", "tenure_years"}.issubset(out.columns):
        out["hours_x_tenure"] = out["weekly_hours"] * out["tenure_years"]
    if {"age_final", "tenure_years"}.issubset(out.columns):
        out["age_x_tenure"] = out["age_final"] * out["tenure_years"]
    if {"age_final", "weekly_hours"}.issubset(out.columns):
        out["age_x_weekly_hours"] = out["age_final"] * out["weekly_hours"]
    if {"monthly_tax_deduction", "monthly_wage"}.issubset(out.columns):
        out["tax_deduction_ratio"] = safe_divide(out["monthly_tax_deduction"], out["monthly_wage"])
    if {"monthly_wage", "starting_monthly_wage"}.issubset(out.columns):
        out["wage_growth_since_start"] = safe_divide(out["monthly_wage"] - out["starting_monthly_wage"], out["starting_monthly_wage"])

    if "firm_size_raw" in out.columns:
        out["one_person_firm_flag"] = (out["firm_size_raw"] == 1).astype(float)

    if {"housing_rent", "household_annual_earned_income"}.issubset(out.columns):
        income_monthly = out["household_annual_earned_income"] / 12.0
        out["housing_cost_burden"] = safe_divide(out["housing_rent"], income_monthly)
    if {"housing_deposit", "household_annual_earned_income"}.issubset(out.columns):
        out["housing_deposit_to_income"] = safe_divide(out["housing_deposit"], out["household_annual_earned_income"])
    if {"household_annual_earned_income", "household_size_raw"}.issubset(out.columns):
        out["household_income_per_capita"] = safe_divide(out["household_annual_earned_income"], out["household_size_raw"])
    if {"household_monthly_savings", "household_monthly_earned_income"}.issubset(out.columns):
        out["savings_to_income"] = safe_divide(out["household_monthly_savings"], out["household_monthly_earned_income"])

    satisfaction_pairs = [
        ("job_sat_wage_raw", "job_sat_wage"),
        ("job_sat_stability_raw", "job_sat_stability"),
        ("job_sat_content_raw", "job_sat_content"),
        ("job_sat_environment_raw", "job_sat_environment"),
        ("job_sat_hours_raw", "job_sat_hours"),
        ("job_sat_development_raw", "job_sat_development"),
        ("job_sat_relationships_raw", "job_sat_relationships"),
        ("job_sat_promotion_raw", "job_sat_promotion"),
        ("job_sat_welfare_raw", "job_sat_welfare"),
        ("job_sat_overall_workplace_raw", "job_sat_overall_workplace"),
        ("job_sat_overall_work_raw", "job_sat_overall_work"),
    ]
    satisfaction_features = []
    for raw_name, feature_name in satisfaction_pairs:
        if raw_name in out.columns:
            out[feature_name] = out[raw_name].copy()
            satisfaction_features.append(feature_name)
    if satisfaction_features:
        out["job_satisfaction_mean"] = out[satisfaction_features].mean(axis=1, skipna=True)
        out["job_satisfaction_missing_count"] = out[satisfaction_features].isna().sum(axis=1)

    categorical_map = {
        "relation_to_head_raw": "relation_to_head",
        "education_level_raw": "education_level",
        "education_completion_raw": "education_completion",
        "education_grade_raw": "education_grade",
        "marital_status_raw": "marital_status",
        "region_raw": "region",
        "industry_raw": "industry_major",
        "occupation_raw": "occupation_major",
        "housing_tenure_raw": "housing_tenure_type",
        "housing_type_raw": "housing_type",
        "health_status_raw": "health_status",
        "health_change_raw": "health_change",
        "employment_type_raw": "employment_type",
        "employee_status_raw": "work_status",
        "work_type_raw": "work_type",
        "job_position_raw": "job_position",
        "regular_worker_raw": "regular_worker",
        "business_place_raw": "business_place",
        "workplace_type_raw": "workplace_type",
        "firm_size_band_raw": "firm_size_band",
        "worksite_size_band_raw": "worksite_size_band",
        "contract_period_raw": "contract_period",
        "written_contract_raw": "written_contract",
        "expected_duration_raw": "expected_duration",
        "voluntary_work_form_raw": "voluntary_work_form",
        "fixed_work_hours_raw": "fixed_work_hours",
        "overtime_work_raw": "overtime_work",
        "pay_schedule_raw": "pay_schedule",
        "pay_method_raw": "pay_method",
        "performance_pay_raw": "performance_pay",
        "national_pension_raw": "national_pension",
        "health_insurance_raw": "health_insurance",
        "employment_insurance_raw": "employment_insurance",
        "industrial_accident_insurance_raw": "industrial_accident_insurance",
        "current_employment_insurance_raw": "current_employment_insurance",
        "union_exists_raw": "union_exists",
        "union_member_raw": "union_member",
        "household_debt_exists_raw": "household_debt_exists",
        "household_financial_condition_raw": "household_financial_condition",
    }
    for raw_name, feature_name in categorical_map.items():
        if raw_name in out.columns:
            out[feature_name] = out[raw_name].astype(object)

    if "survey_year" in out.columns:
        out["survey_year_centered"] = out["survey_year"] - 1998
        out["post_2018_workweek_reform"] = (out["survey_year"] >= 2018).astype(float)
        out["covid_period"] = out["survey_year"].between(2020, 2021).astype(float)
        out["post_covid_period"] = (out["survey_year"] >= 2022).astype(float)

    out = add_panel_history_features(out)

    if "monthly_wage" in out.columns:
        out["wage_percentile_wave"] = out.groupby("wave")["monthly_wage"].rank(pct=True)
        wave_median = out.groupby("wave")["monthly_wage"].transform("median")
        out["wage_to_wave_median"] = safe_divide(out["monthly_wage"], wave_median)
        if "industry_major" in out.columns:
            industry_median = out.groupby(["wave", "industry_major"])["monthly_wage"].transform("median")
            out["wage_to_industry_wave_median"] = safe_divide(out["monthly_wage"], industry_median)
        if "occupation_major" in out.columns:
            occupation_median = out.groupby(["wave", "occupation_major"])["monthly_wage"].transform("median")
            out["wage_to_occupation_wave_median"] = safe_divide(out["monthly_wage"], occupation_median)
    if "weekly_hours" in out.columns:
        out["hours_percentile_wave"] = out.groupby("wave")["weekly_hours"].rank(pct=True)

    out = out.replace({pd.NA: np.nan})
    return out


def make_exit_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = ["pid", "wave", "is_wage_worker_t"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for label generation: {missing}")

    out = df.sort_values(["pid", "wave"]).copy()
                                                                               
                                                                        
                                                                    
    out["wave_t1"] = out.groupby("pid")["wave"].shift(-1)
    out["is_wage_worker_t1"] = out.groupby("pid")["is_wage_worker_t"].shift(-1)
    out["has_next_wave"] = out["wave_t1"] == (out["wave"] + 1)

    n_initial_panel_master = int(len(out))
    wage_risk_mask = out["is_wage_worker_t"] == 1
    n_wage_risk_set = int(wage_risk_mask.sum())
    n_outside_wage_risk_set = int(n_initial_panel_master - n_wage_risk_set)
    valid_follow_mask = wage_risk_mask & out["has_next_wave"]
    n_valid_t1_followup = int(valid_follow_mask.sum())
    n_no_valid_t1_followup = int(n_wage_risk_set - n_valid_t1_followup)

    analysis_base = out[valid_follow_mask].copy()
    analysis_base["exit_label_t1"] = (analysis_base["is_wage_worker_t1"] != 1).astype(int)

    analysis_base.to_csv(PROCESSED_DIR / "analysis_base_with_label.csv", index=False, encoding="utf-8-sig")
    logger.info("Analysis base shape: %s", analysis_base.shape)
    if not analysis_base.empty:
        logger.info("Exit rate: %.4f", analysis_base["exit_label_t1"].mean())

    summary = {
        "initial_panel_master_n": n_initial_panel_master,
        "wage_risk_set_n": n_wage_risk_set,
        "outside_wage_risk_set_n": n_outside_wage_risk_set,
        "valid_t1_followup_n": n_valid_t1_followup,
        "no_valid_t1_followup_n": n_no_valid_t1_followup,
        "analysis_base_n": int(len(analysis_base)),
        "analysis_base_exit_rate": float(analysis_base["exit_label_t1"].mean()) if not analysis_base.empty else None,
        "analysis_base_unique_pid_n": int(analysis_base["pid"].nunique()) if "pid" in analysis_base.columns else None,
    }
    return analysis_base, summary


def fit_baseline_logistic(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols, numeric_features, categorical_features = select_model_columns(train_df)
    target_col = "exit_label_t1"
    if not feature_cols:
        raise ValueError("No feature columns selected.")

    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    y_train = train_df[target_col].astype(int)
    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    y_valid = valid_df[target_col].astype(int)
    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)
    y_test = test_df[target_col].astype(int)

    clf = fit_sklearn_pipeline(
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
        X_train,
        y_train,
        numeric_features,
        categorical_features,
    )

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    valid_metrics = evaluate_binary_classifier(y_valid.to_numpy(), valid_prob)
    test_metrics = evaluate_binary_classifier(y_test.to_numpy(), test_prob)

    rows = []
    for split_name, metric_dict in [("valid", valid_metrics), ("test", test_metrics)]:
        row = {"model": "logistic_baseline", "split": split_name}
        row.update(metric_dict)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "baseline_logistic_metrics.csv", index=False, encoding="utf-8-sig")

    pred_cols = [column for column in ["pid", "wave", target_col] if column in test_df.columns]
    pred_out = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
    pred_out["proba_logistic"] = test_prob
    pred_out.to_csv(OUTPUT_DIR / "baseline_logistic_test_predictions.csv", index=False, encoding="utf-8-sig")

    logger.info("Baseline logistic results saved.")
    return result_df


def write_data_quality_report(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        pd.DataFrame([{"dataset": name, "note": "EMPTY_DATAFRAME"}]).to_csv(
            OUTPUT_DIR / f"dq_{name}.csv", index=False, encoding="utf-8-sig"
        )
        return

    rows = []
    for column in df.columns:
        rows.append(
            {
                "dataset": name,
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_rate": float(df[column].isna().mean()),
                "n_unique": int(df[column].nunique(dropna=True)),
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"dq_{name}.csv", index=False, encoding="utf-8-sig")


def save_path_diagnostics(raw_dir: Path) -> None:
    rows = [
        {"check": "PROJECT_DIR", "value": str(PROJECT_DIR), "exists": PROJECT_DIR.exists()},
        {"check": "RAW_DIR", "value": str(raw_dir), "exists": raw_dir.exists()},
    ]
    preview_files = []
    if raw_dir.exists():
        for index, path in enumerate(raw_dir.rglob("*")):
            if path.is_file():
                preview_files.append(str(path))
            if index >= 49:
                break

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "path_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"preview_file": preview_files}).to_csv(
        OUTPUT_DIR / "path_preview_files.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    logger.info("==== KLIPS SR pipeline start ====")
    logger.info("PROJECT_DIR=%s", PROJECT_DIR)
    logger.info("RAW_DIR=%s", RAW_DIR)

    save_path_diagnostics(RAW_DIR)

    file_metas = discover_klips_files(RAW_DIR)
    if not file_metas:
        raise FileNotFoundError(
            f"No KLIPS files discovered under RAW_DIR={RAW_DIR}. "
            "Check the path, filename pattern, extension (.xlsx/.xls), and whether files are inside subfolders."
        )

    inventory = build_inventory(file_metas)
    logger.info("Inventory preview saved: %s", OUTPUT_DIR / "inventory_preview.csv")
    logger.info("Inventory rows=%s", len(inventory))

    person_df, household_df = load_source_panels(file_metas)
    write_data_quality_report(person_df, "person_extracted")
    write_data_quality_report(household_df, "household_extracted")

    panel_master = build_panel_master(person_df, household_df)
    write_data_quality_report(panel_master, "panel_master")

    core_df = build_core_features(panel_master)
    core_df.to_csv(PROCESSED_DIR / "core_features.csv", index=False, encoding="utf-8-sig")
    write_data_quality_report(core_df, "core_features")
    feature_manifest = build_feature_manifest(set(core_df.columns))
    feature_manifest.to_csv(OUTPUT_DIR / "feature_engineering_manifest.csv", index=False, encoding="utf-8-sig")

    analysis_base, label_summary = make_exit_label(core_df)
    write_data_quality_report(analysis_base, "analysis_base")

    train_df, valid_df, test_df = timewise_split(analysis_base, train_end=20, valid_end=23)

    if min(len(train_df), len(valid_df), len(test_df)) == 0:
        logger.warning("One of the splits is empty. Model training is skipped.")
    else:
        metrics = fit_baseline_logistic(train_df, valid_df, test_df)
        logger.info("\n%s", metrics)

    summary = {
        "project_dir": str(PROJECT_DIR),
        "raw_dir": str(RAW_DIR),
        "n_discovered_files": len(file_metas),
        "person_shape": list(person_df.shape),
        "household_shape": list(household_df.shape),
        "panel_master_shape": list(panel_master.shape),
        "analysis_base_shape": list(analysis_base.shape),
        "analysis_base_exit_rate": float(analysis_base["exit_label_t1"].mean()) if not analysis_base.empty else None,
        "chronological_train_shape": list(train_df.shape),
        "chronological_valid_shape": list(valid_df.shape),
        "chronological_test_shape": list(test_df.shape),
        "model_feature_count": len(select_model_columns(analysis_base)[0]),
        "model_numeric_feature_count": len(select_model_columns(analysis_base)[1]),
        "model_categorical_feature_count": len(select_model_columns(analysis_base)[2]),
    }
    summary.update(label_summary)

    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logger.info("==== KLIPS SR pipeline end ====")


if __name__ == "__main__":
    main()
