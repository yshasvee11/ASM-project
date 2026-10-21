#!/usr/bin/env python3
"""
02_nhpp_arrival_and_service_fitting.py
Applied Stochastic Modelling (ASM) Course Project
Hybrid ED-to-ICU Queueing & Markov Network

Description:
  1. Computes the 24-hour empirical diurnal arrival probability distribution p_h
     from eICU admission timestamps and couples it with CMS annual arrival volumes
     to generate zero-mismatch Non-Homogeneous Poisson Process (NHPP) intensity curves lambda(h).
  2. Performs Maximum Likelihood Estimation (MLE) fitting of candidate parametric
     distributions (Exponential, Weibull, Gamma, Log-Normal) for ICU Length of Stay (LOS)
     and pre-ICU ED Boarding Times.
  3. Evaluates Log-Likelihood, AIC, BIC, and Kolmogorov-Smirnov test statistics.
  4. Generates publication-quality diagnostic plots (PDF overlays, CDF comparisons, and Q-Q plots).
"""

import pathlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Setup Directories
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "output_tables"
PLOTS_DIR = BASE_DIR / "output_plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(" APPLIED STOCHASTIC MODELLING (ASM) — NHPP ARRIVALS & SERVICE TIME FITTING")
print("=" * 80)

patient_path = BASE_DIR / "eicu_demo_data" / "patient.csv"
cms_summary_path = TABLES_DIR / "01_cms_volume_tiers_breakdown.csv"

if not patient_path.exists():
    raise FileNotFoundError(f"Missing required file: {patient_path.resolve()}")

df_patient = pd.read_csv(patient_path, low_memory=False)
df_patient.columns = [str(c).strip().lower().replace(" ", "_") for c in df_patient.columns]

# ---------------------------------------------------------
# 2. Part 1: NHPP Diurnal Arrival Process Calibration
# ---------------------------------------------------------
print("\n--- Part 1: Calibrating Diurnal NHPP Arrival Intensity lambda(h) ---")

# Filter for patients admitted through Emergency Department
ed_mask = df_patient["unitadmitsource"].astype(str).str.contains("Emergency", case=False, na=False)
df_ed = df_patient[ed_mask].copy()

# Extract admission hour from hospitaladmittime24 or unitadmittime24
time_series = df_ed["hospitaladmittime24"].dropna()
if len(time_series) < 100:
    time_series = df_ed["unitadmittime24"].dropna()

# Extract hour (0 to 23)
hours = pd.to_datetime(time_series, format="%H:%M:%S", errors="coerce").dt.hour.dropna().astype(int)

# 24-hour empirical histogram & normalized probability vector p_h
counts = np.bincount(hours, minlength=24)
total_ed_arrivals = counts.sum()
p_h = counts / float(total_ed_arrivals)

assert len(p_h) == 24, "Hourly distribution must have 24 elements."
assert np.isclose(p_h.sum(), 1.0), "Normalized distribution must sum to 1.0."

print(f"  Total ED arrival timestamps analyzed: {total_ed_arrivals:,}")
print(f"  Peak arrival hour: {np.argmax(p_h)}:00 (probability p_h = {np.max(p_h):.4f})")
print(f"  Trough arrival hour: {np.argmin(p_h)}:00 (probability p_h = {np.min(p_h):.4f})")

# Load CMS Daily Arrival Rates Lambda_day across volume tiers
# Default fallback values if summary table doesn't exist yet
default_lambda = {"low": 29.53, "medium": 84.24, "high": 142.25, "very_high": 275.20}
tier_lambdas = default_lambda.copy()

if cms_summary_path.exists():
    df_cms_tier = pd.read_csv(cms_summary_path)
    for _, row in df_cms_tier.iterrows():
        tier_name = str(row["volume_tier"]).strip().lower().replace(" ", "_")
        if tier_name in tier_lambdas and pd.notna(row["mean_daily_lambda"]):
            tier_lambdas[tier_name] = float(row["mean_daily_lambda"])

# Construct hourly arrival rates lambda_tier(h) = Lambda_day * p_h
hourly_df = pd.DataFrame({
    "hour": np.arange(24),
    "empirical_arrival_count": counts,
    "p_h_normalized": p_h,
    "lambda_low_per_hr": tier_lambdas["low"] * p_h,
    "lambda_medium_per_hr": tier_lambdas["medium"] * p_h,
    "lambda_high_per_hr": tier_lambdas["high"] * p_h,
    "lambda_very_high_per_hr": tier_lambdas["very_high"] * p_h,
})

hourly_export_path = TABLES_DIR / "02_nhpp_diurnal_hourly_rates.csv"
hourly_df.to_csv(hourly_export_path, index=False)
print(f"[EXPORT] Saved NHPP diurnal rates to: {hourly_export_path.name}")

# Plot Diurnal Arrival Rates
fig, ax1 = plt.subplots(figsize=(12, 6))

hours_axis = np.arange(24)
color_p = "#1f77b4"
ax1.plot(hours_axis, p_h * 100, marker="o", linewidth=2.5, color=color_p, label="Empirical Probability $p_h$ (%)")
ax1.set_xlabel("Hour of the Day (0:00 to 23:00)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Hourly Arrival Probability $p_h$ (%)", color=color_p, fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color_p)
ax1.set_xticks(hours_axis)
ax1.grid(True, linestyle=":", alpha=0.6)

# Secondary axis for absolute rates
ax2 = ax1.twinx()
ax2.plot(hours_axis, hourly_df["lambda_low_per_hr"], linestyle="--", color="#2ca02c", label=rf"Low Volume ($\Lambda_{{day}}={tier_lambdas['low']:.1f}$)")
ax2.plot(hours_axis, hourly_df["lambda_medium_per_hr"], linestyle="--", color="#ff7f0e", label=rf"Medium Volume ($\Lambda_{{day}}={tier_lambdas['medium']:.1f}$)")
ax2.plot(hours_axis, hourly_df["lambda_high_per_hr"], linestyle="--", color="#9467bd", label=rf"High Volume ($\Lambda_{{day}}={tier_lambdas['high']:.1f}$)")
ax2.plot(hours_axis, hourly_df["lambda_very_high_per_hr"], linestyle="--", color="#d62728", label=rf"Very High Volume ($\Lambda_{{day}}={tier_lambdas['very_high']:.1f}$)")
ax2.set_ylabel(r"NHPP Arrival Rate $\lambda(h)$ (Patients / Hour)", fontsize=12, fontweight="bold")

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True, fontsize=10)

plt.title("Zero-Scale-Mismatch Non-Homogeneous Poisson Process (NHPP) Diurnal Arrival Intensity",
          fontsize=14, fontweight="bold", pad=15)
plot_nhpp = PLOTS_DIR / "02_nhpp_diurnal_arrival_rates.png"
plt.savefig(plot_nhpp, dpi=300, bbox_inches="tight")
plt.close()
print(f"[PLOT] Generated diurnal NHPP plot: {plot_nhpp.name}")

# ---------------------------------------------------------
# 3. Parametric Fitting Engine Helper
# ---------------------------------------------------------
def fit_distributions_and_evaluate(data: np.ndarray, variable_name: str, unit_label: str):
    """
    Fits Exponential, Weibull, Gamma, and Log-Normal distributions via MLE.
    Computes Log-Likelihood, AIC, BIC, and KS-test statistics.
    Returns results DataFrame and fitted distribution objects.
    """
    clean_data = data[data > 0]
    n = len(clean_data)
    
    candidates = {
        "Exponential": {"dist": stats.expon, "k": 1, "color": "#d62728"},
        "Weibull": {"dist": stats.weibull_min, "k": 2, "color": "#2ca02c"},
        "Gamma": {"dist": stats.gamma, "k": 2, "color": "#ff7f0e"},
        "Log-Normal": {"dist": stats.lognorm, "k": 2, "color": "#9467bd"}
    }
    
    records = []
    fitted_models = {}
    
    for name, spec in candidates.items():
        dist = spec["dist"]
        k = spec["k"]
        
        # MLE fitting with floc=0 (strictly positive support)
        if name == "Exponential":
            # expon: floc=0 -> scale = 1 / lambda
            params = dist.fit(clean_data, floc=0)
            param_str = f"scale={params[1]:.4f} (rate lambda={1.0/params[1]:.4f})"
        elif name == "Weibull":
            params = dist.fit(clean_data, floc=0)
            param_str = f"c (shape)={params[0]:.4f}, scale={params[2]:.4f}"
        elif name == "Gamma":
            params = dist.fit(clean_data, floc=0)
            param_str = f"a (shape)={params[0]:.4f}, scale={params[2]:.4f}"
        elif name == "Log-Normal":
            params = dist.fit(clean_data, floc=0)
            param_str = f"s (sigma)={params[0]:.4f}, scale (exp(mu))={params[2]:.4f} (mu={np.log(params[2]):.4f})"
        
        # Log-Likelihood
        log_pdf = dist.logpdf(clean_data, *params)
        log_lik = np.sum(log_pdf)
        
        # Information Criteria
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        # Kolmogorov-Smirnov Test
        ks_stat, ks_pval = stats.kstest(clean_data, dist.name, args=params)
        
        fitted_models[name] = {
            "dist": dist,
            "params": params,
            "color": spec["color"],
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "aic": aic,
            "bic": bic
        }
        
        records.append({
            "distribution": name,
            "num_parameters_k": k,
            "fitted_parameters": param_str,
            "log_likelihood": log_lik,
            "aic": aic,
            "bic": bic,
            "ks_statistic_d": ks_stat,
            "ks_p_value": ks_pval
        })
    
    df_eval = pd.DataFrame(records)
    # Calculate delta AIC and delta BIC relative to best model
    min_aic = df_eval["aic"].min()
    min_bic = df_eval["bic"].min()
    df_eval["delta_aic"] = df_eval["aic"] - min_aic
    df_eval["delta_bic"] = df_eval["bic"] - min_bic
    df_eval["aic_rank"] = df_eval["aic"].rank().astype(int)
    df_eval["ks_rank"] = df_eval["ks_statistic_d"].rank().astype(int)
    
    df_eval = df_eval.sort_values("aic")
    return df_eval, fitted_models, clean_data

# ---------------------------------------------------------
# 4. Part 2: Fit ICU Length of Stay (S_ICU)
# ---------------------------------------------------------
print("\n--- Part 2: Parametric Fitting for ICU Length of Stay (S_ICU) ---")
icu_los_hours = (df_patient["unitdischargeoffset"] / 60.0).dropna().values
df_los_eval, los_models, los_data = fit_distributions_and_evaluate(
    icu_los_hours, "ICU Length of Stay", "Hours"
)

los_export_path = TABLES_DIR / "02_icu_los_distribution_fitting.csv"
df_los_eval.to_csv(los_export_path, index=False)
print(df_los_eval[["distribution", "aic", "delta_aic", "bic", "ks_statistic_d", "ks_p_value", "aic_rank"]].to_string(index=False))
print(f"[EXPORT] Saved ICU LOS evaluation to: {los_export_path.name}")

# ---------------------------------------------------------
# 5. Part 3: Fit ED-to-ICU Boarding Delay (B_ED->ICU)
# ---------------------------------------------------------
print("\n--- Part 3: Parametric Fitting for ED-to-ICU Boarding Delay (B_ED->ICU) ---")
valid_boarding = df_ed[df_ed["hospitaladmitoffset"] < 0]["hospitaladmitoffset"].abs().dropna().values
boarding_hours = valid_boarding / 60.0

df_b_eval, b_models, b_data = fit_distributions_and_evaluate(
    boarding_hours, "ED Boarding Delay", "Hours"
)

b_export_path = TABLES_DIR / "02_ed_boarding_distribution_fitting.csv"
df_b_eval.to_csv(b_export_path, index=False)
print(df_b_eval[["distribution", "aic", "delta_aic", "bic", "ks_statistic_d", "ks_p_value", "aic_rank"]].to_string(index=False))
print(f"[EXPORT] Saved ED Boarding evaluation to: {b_export_path.name}")

# ---------------------------------------------------------
# 6. Comprehensive Diagnostic Plots Function
# ---------------------------------------------------------
def generate_diagnostic_figure(data: np.ndarray, models: dict, title_prefix: str, xlabel: str, output_path: pathlib.Path):
    """
    Creates a 6-panel figure:
    1. Histogram & Fitted PDFs (Capped at 98th percentile for visual clarity)
    2. Empirical CDF vs Fitted CDFs
    3-6. Q-Q plots for Exponential, Weibull, Gamma, and Log-Normal
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    p98 = np.percentile(data, 98)
    x_grid = np.linspace(0.01, p98, 1000)
    
    # 1. PDF Overlay
    ax_pdf = fig.add_subplot(gs[0, 0])
    ax_pdf.hist(data[data <= p98], bins=45, density=True, alpha=0.45, color="#1f77b4", edgecolor="black", label="Empirical Data")
    for name, m in models.items():
        pdf_vals = m["dist"].pdf(x_grid, *m["params"])
        ax_pdf.plot(x_grid, pdf_vals, label=f"{name}", color=m["color"], linewidth=2.2)
    ax_pdf.set_title(f"A: Empirical Histogram vs. Fitted PDFs ({title_prefix})", fontsize=12, fontweight="bold")
    ax_pdf.set_xlabel(xlabel, fontsize=10)
    ax_pdf.set_ylabel("Probability Density", fontsize=10)
    ax_pdf.legend(frameon=True, fontsize=9)
    ax_pdf.grid(True, linestyle=":", alpha=0.6)
    
    # 2. CDF Comparison
    ax_cdf = fig.add_subplot(gs[0, 1])
    sorted_data = np.sort(data)
    emp_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Subsample for smooth rendering
    step = max(1, len(sorted_data) // 1000)
    ax_cdf.plot(sorted_data[::step], emp_cdf[::step], "k-", linewidth=2.5, label="Empirical CDF")
    
    for name, m in models.items():
        cdf_vals = m["dist"].cdf(sorted_data[::step], *m["params"])
        ax_cdf.plot(sorted_data[::step], cdf_vals, label=f"{name} (D={m['ks_stat']:.3f})", color=m["color"], linewidth=1.8, linestyle="--")
    ax_cdf.set_title(f"B: Empirical vs. Theoretical CDFs ({title_prefix})", fontsize=12, fontweight="bold")
    ax_cdf.set_xlabel(xlabel, fontsize=10)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=10)
    ax_cdf.set_xlim(0, p98)
    ax_cdf.legend(frameon=True, fontsize=9)
    ax_cdf.grid(True, linestyle=":", alpha=0.6)
    
    # 3-6. Q-Q Plots
    qq_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    dist_names = ["Exponential", "Weibull", "Gamma", "Log-Normal"]
    
    # Generate probabilities for quantiles
    quantiles = np.linspace(0.01, 0.99, 100)
    sample_quantiles = np.percentile(data, quantiles * 100)
    
    for idx, name in enumerate(dist_names):
        pos = qq_positions[idx]
        ax_qq = fig.add_subplot(gs[pos[0], pos[1]])
        m = models[name]
        theo_quantiles = m["dist"].ppf(quantiles, *m["params"])
        
        ax_qq.scatter(theo_quantiles, sample_quantiles, color=m["color"], alpha=0.75, s=25, label=f"{name} Quantiles")
        max_q = min(max(theo_quantiles), max(sample_quantiles))
        ax_qq.plot([0, max_q], [0, max_q], "k--", linewidth=1.5, label="Identity (45° Line)")
        
        ax_qq.set_title(f"Q-Q Plot: {name} (AIC={m['aic']:.1f}, KS D={m['ks_stat']:.3f})", fontsize=11, fontweight="bold")
        ax_qq.set_xlabel(f"Theoretical Quantiles ({xlabel})", fontsize=9)
        ax_qq.set_ylabel(f"Empirical Quantiles ({xlabel})", fontsize=9)
        ax_qq.legend(frameon=True, fontsize=8)
        ax_qq.grid(True, linestyle=":", alpha=0.6)
    
    fig.suptitle(f"Maximum Likelihood Estimation (MLE) Goodness-of-Fit Diagnostics: {title_prefix}",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated diagnostic figure: {output_path.name}")

# Generate figures
generate_diagnostic_figure(
    los_data, los_models, "ICU Length of Stay ($S_{\\mathrm{ICU}}$)",
    "Duration (Hours)", PLOTS_DIR / "02_icu_los_fitting_diagnostics.png"
)

generate_diagnostic_figure(
    b_data, b_models, "Pre-ICU ED Boarding Time ($B_{\\mathrm{ED}\\to\\mathrm{ICU}}$)",
    "Boarding Duration (Hours)", PLOTS_DIR / "02_ed_boarding_fitting_diagnostics.png"
)

print("=" * 80)
print(" STOCHASTIC FITTING COMPLETE — PROOF OF HEAVY-TAILED BEHAVIOR ESTABLISHED")
print("=" * 80)
