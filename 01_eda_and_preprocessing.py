#!/usr/bin/env python3
"""
01_eda_and_preprocessing.py
Applied Stochastic Modelling (ASM) Course Project
Hybrid ED-to-ICU Queueing & Markov Network

Description:
  Loads and normalizes micro patient-level data (eICU-CRD Demo v2.0.1) and
  macro facility-level benchmarks (CMS Timely and Effective Care yv7e-xc69).
  Computes cohort distributions, offsets, hospital bed categories, CMS ED
  volume tiers, and calibrates parameters for subsequent queueing and Markov analysis.
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Directory Structure Setup
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "output_tables"
PLOTS_DIR = BASE_DIR / "output_plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Schema Normalization Helper
# ---------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase stripped strings."""
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

print("=" * 80)
print(" APPLIED STOCHASTIC MODELLING (ASM) — DATA PREPROCESSING & EDA")
print("=" * 80)

# Paths
patient_path = BASE_DIR / "eicu_demo_data" / "patient.csv"
hospital_path = BASE_DIR / "eicu_demo_data" / "hospital.csv"
cms_path = BASE_DIR / "cms_data" / "cms_timely_and_effective_care_hospital.csv"

# Validate paths
for p in [patient_path, hospital_path, cms_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required data file: {p.resolve()}")

print(f"[LOAD] Loading eICU Patient data from: {patient_path.name}")
df_patient = normalize_dataframe(pd.read_csv(patient_path, low_memory=False))

print(f"[LOAD] Loading eICU Hospital data from: {hospital_path.name}")
df_hospital = normalize_dataframe(pd.read_csv(hospital_path, low_memory=False))

print(f"[LOAD] Loading CMS Hospital Care data from: {cms_path.name}")
df_cms = normalize_dataframe(pd.read_csv(cms_path, low_memory=False))

# Handle possible CMS column aliases
if "_condition" in df_cms.columns and "condition" not in df_cms.columns:
    df_cms["condition"] = df_cms["_condition"]

# ---------------------------------------------------------
# 3. Micro Patient-Level Exploratory Data Analysis (eICU)
# ---------------------------------------------------------
total_stays = len(df_patient)
unique_patients = df_patient["uniquepid"].nunique() if "uniquepid" in df_patient.columns else np.nan
unique_hosp_stays = df_patient["patienthealthsystemstayid"].nunique() if "patienthealthsystemstayid" in df_patient.columns else np.nan
hospitals_represented = df_patient["hospitalid"].nunique()

print("\n--- eICU Cohort Overview ---")
print(f"  Total ICU Encounters (stays):       {total_stays:,}")
print(f"  Unique Patient Health System Stays: {unique_hosp_stays:,}")
print(f"  Unique Patients (uniquepid):        {unique_patients:,}")
print(f"  Hospitals Represented in Stays:     {hospitals_represented}")

# Mortality Rates
unit_mortality = (df_patient["unitdischargestatus"].str.lower() == "expired").mean() * 100
hosp_mortality = (df_patient["hospitaldischargestatus"].str.lower() == "expired").mean() * 100
print(f"  ICU Discharge Mortality Rate:       {unit_mortality:.2f}%")
print(f"  Hospital Discharge Mortality Rate:  {hosp_mortality:.2f}%")

# Emergency Department Admitted Cohort
ed_mask = df_patient["unitadmitsource"].astype(str).str.contains("Emergency", case=False, na=False)
df_ed = df_patient[ed_mask].copy()
ed_stays = len(df_ed)
ed_fraction = (ed_stays / total_stays) * 100
print(f"\n--- Upstream Emergency Department Subset ---")
print(f"  Patients Admitted to ICU via ED:    {ed_stays:,} ({ed_fraction:.2f}% of all ICU stays)")

# Pre-ICU Boarding Offset ($B_{\text{ED}\to\text{ICU}} = -\text{hospitaladmitoffset}$)
# Negative offset means hospital admission happened prior to ICU admission
valid_boarding = df_ed[df_ed["hospitaladmitoffset"] < 0]["hospitaladmitoffset"].abs()
print(f"  Valid ED Boarding Records:          {len(valid_boarding):,} patients")
print(f"  ED-to-ICU Boarding Time (Mean):     {valid_boarding.mean():.1f} minutes ({valid_boarding.mean()/60:.2f} hours)")
print(f"  ED-to-ICU Boarding Time (Median):   {valid_boarding.median():.1f} minutes ({valid_boarding.median()/60:.2f} hours)")
print(f"  ED-to-ICU Boarding Time (IQR):      {valid_boarding.quantile(0.25):.1f} to {valid_boarding.quantile(0.75):.1f} minutes")

# ICU Length of Stay ($S_{\text{ICU}} = \text{unitdischargeoffset}$)
icu_los_hours = df_patient["unitdischargeoffset"] / 60.0
icu_los_days = icu_los_hours / 24.0
print(f"\n--- Downstream ICU Length of Stay (LOS) ---")
print(f"  Mean ICU LOS:                       {icu_los_hours.mean():.2f} hours ({icu_los_days.mean():.2f} days)")
print(f"  Median ICU LOS:                     {icu_los_hours.median():.2f} hours ({icu_los_days.median():.2f} days)")
print(f"  Standard Deviation:                 {icu_los_hours.std():.2f} hours")
print(f"  Coefficient of Variation (std/mean): {icu_los_hours.std() / icu_los_hours.mean():.3f} (Sign of non-exponentiality!)")

# ---------------------------------------------------------
# 4. Hospital Capacity & Routing Distributions
# ---------------------------------------------------------
# Unit Types
unit_type_counts = df_patient["unittype"].value_counts().reset_index()
unit_type_counts.columns = ["unit_type", "patient_count"]
unit_type_counts["percentage"] = (unit_type_counts["patient_count"] / total_stays) * 100

# Hospital Bed Tiers
df_hosp_merged = df_patient.merge(df_hospital, on="hospitalid", how="left")
bed_tier_counts = df_hosp_merged["numbedscategory"].value_counts(dropna=False).reset_index()
bed_tier_counts.columns = ["bed_capacity_category", "icu_stays"]
bed_tier_counts["percentage"] = (bed_tier_counts["icu_stays"] / total_stays) * 100

# Admission Sources
adm_source_counts = df_patient["unitadmitsource"].value_counts(dropna=False).reset_index()
adm_source_counts.columns = ["unit_admit_source", "patient_count"]
adm_source_counts["percentage"] = (adm_source_counts["patient_count"] / total_stays) * 100

# Discharge Locations
disch_loc_counts = df_patient["unitdischargelocation"].value_counts(dropna=False).reset_index()
disch_loc_counts.columns = ["unit_discharge_location", "patient_count"]
disch_loc_counts["percentage"] = (disch_loc_counts["patient_count"] / total_stays) * 100

# ---------------------------------------------------------
# 5. Macro Facility-Level Queueing Benchmarks (CMS)
# ---------------------------------------------------------
print("\n--- CMS Timely and Effective Care Benchmarks ---")
# Filter Emergency Department Condition
cms_ed = df_cms[df_cms["condition"].astype(str).str.contains("Emergency", case=False, na=False)].copy()

# Convert score and sample to numeric
cms_ed["score_num"] = pd.to_numeric(cms_ed["score"], errors="coerce")
cms_ed["sample_num"] = pd.to_numeric(cms_ed["sample"], errors="coerce")

# Pivot facility-level measures
measures_of_interest = ["EDV", "OP_22", "OP_18a", "OP_18b"]
df_facility_measures = cms_ed[cms_ed["measure_id"].isin(measures_of_interest)].copy()

# Extract per-facility metrics
facility_piv = df_facility_measures.pivot_table(
    index=["facility_id", "facility_name"],
    columns="measure_id",
    values=["score", "score_num", "sample_num"],
    aggfunc="first"
)

# Build clean CMS summary per facility
facilities = []
for fid, group in df_facility_measures.groupby("facility_id"):
    name = group["facility_name"].iloc[0]
    edv = group.loc[group["measure_id"] == "EDV", "score"].values
    edv_val = edv[0].strip().lower() if len(edv) > 0 and pd.notna(edv[0]) else "unknown"
    
    op22_row = group[group["measure_id"] == "OP_22"]
    annual_arrivals = op22_row["sample_num"].values[0] if len(op22_row) > 0 else np.nan
    abandon_rate = op22_row["score_num"].values[0] if len(op22_row) > 0 else np.nan
    
    op18b_row = group[group["measure_id"] == "OP_18b"]
    ed_sojourn_min = op18b_row["score_num"].values[0] if len(op18b_row) > 0 else np.nan
    
    facilities.append({
        "facility_id": fid,
        "facility_name": name,
        "volume_tier": edv_val,
        "annual_arrivals_n": annual_arrivals,
        "daily_arrival_rate_lambda": annual_arrivals / 365.0 if pd.notna(annual_arrivals) else np.nan,
        "abandonment_pct_p_abandon": abandon_rate,
        "median_ed_sojourn_minutes_w_ed": ed_sojourn_min
    })

df_cms_facilities = pd.DataFrame(facilities)

# Tier-level aggregation
tier_stats = df_cms_facilities.groupby("volume_tier").agg(
    facility_count=("facility_id", "count"),
    mean_annual_arrivals=("annual_arrivals_n", "mean"),
    median_annual_arrivals=("annual_arrivals_n", "median"),
    mean_daily_lambda=("daily_arrival_rate_lambda", "mean"),
    median_daily_lambda=("daily_arrival_rate_lambda", "median"),
    mean_abandonment_pct=("abandonment_pct_p_abandon", "mean"),
    mean_ed_sojourn_min=("median_ed_sojourn_minutes_w_ed", "mean"),
    median_ed_sojourn_min=("median_ed_sojourn_minutes_w_ed", "median")
).reset_index()

# Sort volume tiers logically
tier_order = {"low": 1, "medium": 2, "high": 3, "very high": 4, "unknown": 5}
tier_stats["sort_idx"] = tier_stats["volume_tier"].map(tier_order).fillna(99)
tier_stats = tier_stats.sort_values("sort_idx").drop(columns=["sort_idx"])

print(tier_stats.to_string(index=False))

# ---------------------------------------------------------
# 6. Save Normalized Tables to output_tables/
# ---------------------------------------------------------
summary_overview = pd.DataFrame([{
    "total_icu_stays": total_stays,
    "unique_patients": unique_patients,
    "unique_hospital_stays": unique_hosp_stays,
    "ed_admitted_icu_stays": ed_stays,
    "ed_admitted_percentage": ed_fraction,
    "icu_mortality_pct": unit_mortality,
    "hospital_mortality_pct": hosp_mortality,
    "mean_icu_los_hours": icu_los_hours.mean(),
    "median_icu_los_hours": icu_los_hours.median(),
    "std_icu_los_hours": icu_los_hours.std(),
    "mean_ed_boarding_minutes": valid_boarding.mean(),
    "median_ed_boarding_minutes": valid_boarding.median()
}])

summary_overview.to_csv(TABLES_DIR / "01_overall_cohort_overview.csv", index=False)
unit_type_counts.to_csv(TABLES_DIR / "01_unit_type_distribution.csv", index=False)
bed_tier_counts.to_csv(TABLES_DIR / "01_bed_tier_distribution.csv", index=False)
adm_source_counts.to_csv(TABLES_DIR / "01_admission_source_distribution.csv", index=False)
disch_loc_counts.to_csv(TABLES_DIR / "01_discharge_location_distribution.csv", index=False)
df_cms_facilities.to_csv(TABLES_DIR / "01_cms_facilities_benchmarks.csv", index=False)
tier_stats.to_csv(TABLES_DIR / "01_cms_volume_tiers_breakdown.csv", index=False)

print("\n[EXPORT] Successfully saved summary tables to 'output_tables/':")
print("  - 01_overall_cohort_overview.csv")
print("  - 01_unit_type_distribution.csv")
print("  - 01_bed_tier_distribution.csv")
print("  - 01_admission_source_distribution.csv")
print("  - 01_discharge_location_distribution.csv")
print("  - 01_cms_facilities_benchmarks.csv")
print("  - 01_cms_volume_tiers_breakdown.csv")

# ---------------------------------------------------------
# 7. Exploratory Visualizations
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Plot A: ICU Length of Stay Distribution (Capped at 99th percentile for visual clarity)
los_clean = icu_los_hours.dropna()
p99 = np.percentile(los_clean, 99)
axs[0, 0].hist(los_clean[los_clean <= p99], bins=45, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
axs[0, 0].axvline(los_clean.mean(), color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {los_clean.mean():.1f}h")
axs[0, 0].axvline(los_clean.median(), color="#2ca02c", linestyle="-", linewidth=2, label=f"Median: {los_clean.median():.1f}h")
axs[0, 0].set_title("A: Empirical ICU Length of Stay Distribution (LOS)", fontsize=13, fontweight="bold")
axs[0, 0].set_xlabel("ICU Length of Stay (Hours)", fontsize=11)
axs[0, 0].set_ylabel("Probability Density", fontsize=11)
axs[0, 0].legend(frameon=True)
axs[0, 0].grid(True, linestyle=":", alpha=0.6)

# Plot B: Pre-ICU ED Boarding Delay Distribution
b_p98 = np.percentile(valid_boarding, 98)
axs[0, 1].hist(valid_boarding[valid_boarding <= b_p98], bins=40, color="#ff7f0e", edgecolor="black", alpha=0.75, density=True)
axs[0, 1].axvline(valid_boarding.mean(), color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {valid_boarding.mean():.0f} min")
axs[0, 1].axvline(valid_boarding.median(), color="#2ca02c", linestyle="-", linewidth=2, label=f"Median: {valid_boarding.median():.0f} min")
axs[0, 1].set_title(r"B: Empirical Pre-ICU Boarding Delay ($B_{\mathrm{ED}\to\mathrm{ICU}}$)", fontsize=13, fontweight="bold")
axs[0, 1].set_xlabel("Boarding Duration in ED (Minutes)", fontsize=11)
axs[0, 1].set_ylabel("Probability Density", fontsize=11)
axs[0, 1].legend(frameon=True)
axs[0, 1].grid(True, linestyle=":", alpha=0.6)

# Plot C: Admission Sources Bar Chart
top_adm = adm_source_counts.head(7).copy()
axs[1, 0].barh(top_adm["unit_admit_source"], top_adm["patient_count"], color="#2ca02c", edgecolor="black", alpha=0.8)
axs[1, 0].invert_yaxis()
for i, v in enumerate(top_adm["patient_count"]):
    axs[1, 0].text(v + 15, i, f"{v:,} ({top_adm['percentage'].iloc[i]:.1f}%)", va="center", fontsize=9)
axs[1, 0].set_title("C: Patient Inflow Sources to ICU", fontsize=13, fontweight="bold")
axs[1, 0].set_xlabel("Number of ICU Encounters", fontsize=11)
axs[1, 0].grid(axis="x", linestyle=":", alpha=0.6)

# Plot D: CMS ED Volume vs LWBS Abandonment Rate
valid_cms = df_cms_facilities.dropna(subset=["annual_arrivals_n", "abandonment_pct_p_abandon"])
color_map = {"low": "#2ca02c", "medium": "#1f77b4", "high": "#ff7f0e", "very high": "#d62728", "unknown": "#7f7f7f"}
for tier, grp in valid_cms.groupby("volume_tier"):
    axs[1, 1].scatter(grp["annual_arrivals_n"] / 1000.0, grp["abandonment_pct_p_abandon"],
                      label=tier.title(), color=color_map.get(tier, "blue"), s=60, alpha=0.8, edgecolors="black")

axs[1, 1].set_title(r"D: CMS ED Annual Arrivals vs. Abandonment Rate ($P_{\mathrm{abandon}}$)", fontsize=13, fontweight="bold")
axs[1, 1].set_xlabel("Annual ED Patient Volume (Thousands)", fontsize=11)
axs[1, 1].set_ylabel("Left Without Being Seen (LWBS %)", fontsize=11)
axs[1, 1].legend(title="ED Volume Tier", frameon=True)
axs[1, 1].grid(True, linestyle=":", alpha=0.6)

fig.suptitle("Applied Stochastic Modelling (ASM): Exploratory Data Analysis & Empirical Calibration",
             fontsize=16, fontweight="bold", y=0.99)
plot_filename = PLOTS_DIR / "01_eda_distributions.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"[PLOT] Generated publication-quality figure: {plot_filename.resolve()}")
print("=" * 80)
print(" PREPROCESSING & EDA COMPLETE — READY FOR STOCHASTIC FITTING (SCRIPT 02)")
print("=" * 80)
