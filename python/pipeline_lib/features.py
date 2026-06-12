from __future__ import annotations

from typing import List, Tuple

import pandas as pd


BASELINE_FEATURES = [
    "gender",
    "age_final",
    "education_level",
    "marital_status",
    "region",
    "industry_major",
    "occupation_major",
    "tenure_years",
    "weekly_hours",
    "weekly_hours_missing",
    "gt48",
    "gt52",
    "ge55",
    "monthly_wage",
    "monthly_wage_missing",
    "log_monthly_wage",
    "one_person_firm_flag",
]


EXPANDED_FEATURES = [
                                                
    "relation_to_head",
    "education_completion",
    "education_grade",
    "health_status",
    "health_change",
    "household_size_raw",
    "housing_tenure_type",
    "housing_type",
    "housing_deposit",
    "housing_rent",
    "household_annual_earned_income",
    "household_monthly_earned_income",
    "household_medical_cost",
    "household_monthly_savings",
    "household_bank_savings",
    "household_debt_exists",
    "household_financial_condition",
    "household_income_per_capita",
    "housing_cost_burden",
    "housing_deposit_to_income",
    "savings_to_income",
                                              
    "employment_type",
    "work_status",
    "work_type",
    "job_position",
    "regular_worker",
    "business_place",
    "workplace_type",
    "firm_size_band",
    "worksite_size",
    "worksite_size_band",
    "contract_period",
    "written_contract",
    "expected_duration",
    "voluntary_work_form",
                            
    "fixed_work_hours",
    "weekly_work_days",
    "regular_weekly_hours",
    "regular_weekly_days",
    "overtime_work",
    "overtime_hours",
    "overtime_days",
    "overtime_pay",
                               
    "pay_schedule",
    "pay_method",
    "performance_pay",
    "monthly_tax_deduction",
    "starting_monthly_wage",
    "annual_earned_income",
    "aftertax_earned_income",
    "national_pension",
    "health_insurance",
    "employment_insurance",
    "industrial_accident_insurance",
    "current_employment_insurance",
    "union_exists",
    "union_member",
    "log_annual_earned_income",
    "log_aftertax_earned_income",
    "log_starting_monthly_wage",
    "hourly_wage_proxy",
    "log_hourly_wage_proxy",
    "tax_deduction_ratio",
    "wage_growth_since_start",
                                        
    "job_sat_wage",
    "job_sat_stability",
    "job_sat_content",
    "job_sat_environment",
    "job_sat_hours",
    "job_sat_development",
    "job_sat_relationships",
    "job_sat_promotion",
    "job_sat_welfare",
    "job_sat_overall_workplace",
    "job_sat_overall_work",
    "job_satisfaction_mean",
    "job_satisfaction_missing_count",
                                        
    "age_squared",
    "tenure_squared",
    "log_tenure_years",
    "age_x_tenure",
    "age_x_weekly_hours",
    "wage_x_tenure",
    "wage_x_hours",
    "hours_x_tenure",
    "young_worker_flag",
    "older_worker_flag",
    "short_tenure_flag",
    "long_tenure_flag",
                                           
    "survey_year_centered",
    "post_2018_workweek_reform",
    "covid_period",
    "post_covid_period",
    "panel_observation_number",
    "prior_observation_count",
    "waves_since_first_observed",
    "prior_wage_worker_count",
    "prior_non_wage_worker_count",
    "prior_wage_worker_share",
    "monthly_wage_lag1",
    "weekly_hours_lag1",
    "tenure_years_lag1",
    "firm_size_lag1",
    "wage_change_lag1",
    "wage_pct_change_lag1",
    "hours_change_lag1",
    "hours_pct_change_lag1",
    "tenure_change_lag1",
    "wage_drop_flag",
    "hours_drop_flag",
    "occupation_changed_lag1",
    "industry_changed_lag1",
    "region_changed_lag1",
    "firm_size_changed_lag1",
    "wage_rolling3_mean",
    "wage_rolling3_std",
    "hours_rolling3_mean",
    "hours_rolling3_std",
    "wage_percentile_wave",
    "hours_percentile_wave",
    "wage_to_wave_median",
    "wage_to_industry_wave_median",
    "wage_to_occupation_wave_median",
]


MODEL_FEATURES = BASELINE_FEATURES + EXPANDED_FEATURES


NUMERIC_FEATURES = {
    "age_final",
    "tenure_years",
    "weekly_hours",
    "weekly_hours_missing",
    "gt48",
    "gt52",
    "ge55",
    "monthly_wage",
    "monthly_wage_missing",
    "log_monthly_wage",
    "one_person_firm_flag",
    "household_size_raw",
    "housing_deposit",
    "housing_rent",
    "household_annual_earned_income",
    "household_monthly_earned_income",
    "household_medical_cost",
    "household_monthly_savings",
    "household_bank_savings",
    "household_income_per_capita",
    "housing_cost_burden",
    "housing_deposit_to_income",
    "savings_to_income",
    "worksite_size",
    "weekly_work_days",
    "regular_weekly_hours",
    "regular_weekly_days",
    "overtime_hours",
    "overtime_days",
    "overtime_pay",
    "monthly_tax_deduction",
    "starting_monthly_wage",
    "annual_earned_income",
    "aftertax_earned_income",
    "log_annual_earned_income",
    "log_aftertax_earned_income",
    "log_starting_monthly_wage",
    "hourly_wage_proxy",
    "log_hourly_wage_proxy",
    "tax_deduction_ratio",
    "wage_growth_since_start",
    "job_sat_wage",
    "job_sat_stability",
    "job_sat_content",
    "job_sat_environment",
    "job_sat_hours",
    "job_sat_development",
    "job_sat_relationships",
    "job_sat_promotion",
    "job_sat_welfare",
    "job_sat_overall_workplace",
    "job_sat_overall_work",
    "job_satisfaction_mean",
    "job_satisfaction_missing_count",
    "age_squared",
    "tenure_squared",
    "log_tenure_years",
    "age_x_tenure",
    "age_x_weekly_hours",
    "wage_x_tenure",
    "wage_x_hours",
    "hours_x_tenure",
    "young_worker_flag",
    "older_worker_flag",
    "short_tenure_flag",
    "long_tenure_flag",
    "survey_year_centered",
    "post_2018_workweek_reform",
    "covid_period",
    "post_covid_period",
    "panel_observation_number",
    "prior_observation_count",
    "waves_since_first_observed",
    "prior_wage_worker_count",
    "prior_non_wage_worker_count",
    "prior_wage_worker_share",
    "monthly_wage_lag1",
    "weekly_hours_lag1",
    "tenure_years_lag1",
    "firm_size_lag1",
    "wage_change_lag1",
    "wage_pct_change_lag1",
    "hours_change_lag1",
    "hours_pct_change_lag1",
    "tenure_change_lag1",
    "wage_drop_flag",
    "hours_drop_flag",
    "occupation_changed_lag1",
    "industry_changed_lag1",
    "region_changed_lag1",
    "firm_size_changed_lag1",
    "wage_rolling3_mean",
    "wage_rolling3_std",
    "hours_rolling3_mean",
    "hours_rolling3_std",
    "wage_percentile_wave",
    "hours_percentile_wave",
    "wage_to_wave_median",
    "wage_to_industry_wave_median",
    "wage_to_occupation_wave_median",
}


_INTERACTION_START = EXPANDED_FEATURES.index("age_squared")
_PANEL_START = EXPANDED_FEATURES.index("survey_year_centered")

FEATURE_GROUPS = {
    "baseline_harmonised": BASELINE_FEATURES,
    "expanded_raw_questionnaire": EXPANDED_FEATURES[:_INTERACTION_START],
    "nonlinear_interactions": EXPANDED_FEATURES[_INTERACTION_START:_PANEL_START],
    "time_aware_panel_features": EXPANDED_FEATURES[_PANEL_START:],
}


MIN_OBSERVED_FEATURE_VALUES = 20


def select_model_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    candidate_features = []
    for column in MODEL_FEATURES:
        if column not in df.columns:
            continue
        if int(df[column].notna().sum()) < MIN_OBSERVED_FEATURE_VALUES:
            continue
        candidate_features.append(column)
    numeric_features = [column for column in candidate_features if column in NUMERIC_FEATURES]
    categorical_features = [column for column in candidate_features if column not in NUMERIC_FEATURES]
    return candidate_features, numeric_features, categorical_features


def build_feature_manifest(available_columns: set[str] | None = None) -> pd.DataFrame:
    available_columns = available_columns or set(MODEL_FEATURES)
    rows = []
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            rows.append(
                {
                    "feature_group": group,
                    "feature": feature,
                    "model_type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
                    "available_in_analysis_file": feature in available_columns,
                }
            )
    return pd.DataFrame(rows)
