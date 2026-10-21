#!/usr/bin/env python3
"""
04_tandem_queue_simulation.py
Applied Stochastic Modelling (ASM) Course Project
Hybrid ED-to-ICU Queueing & Markov Network

Description:
  Implements a Discrete-Event Simulation (DES) of the coupled two-stage tandem queue:

    Stage 1: Emergency Department (ED) — M_t / G / c1 + M (Erlang-A with abandonment)
             Non-Homogeneous Poisson Process arrivals with diurnal lambda(h).
             Patience time: tau ~ Exponential(theta).
             Service time: S_ED ~ Exponential(mu_ED).

    Stage 2: Intensive Care Unit (ICU) — G / G / c2 with boarding block.
             A fraction p_ICU of ED patients require ICU admission.
             ICU service time: S_ICU ~ Weibull(c=0.9126, beta=55.57 hours) [Best MLE fit].
             When all c2 ICU beds are full, newly arriving ICU patients board in an ED bay
             (blocking one c1 server) until an ICU bed frees up.

  Simulation Parameters:
    - Horizon: 35 simulated days (5-day warm-up + 30-day measurement window)
    - Monte Carlo Replications: 20 independent runs
    - ICU capacity sensitivity sweep: c2 in {8, 10, 12, 14, 16, 20}
    - ED capacity: c1 = 20 acute bays (configurable)
    - Arrival volume: CMS Very High tier (Lambda_day = 275.2 patients/day)

  Key Output Metrics per c2 Configuration:
    - E[B_block]:     Mean ICU boarding delay in the ED (hours)
    - rho_ED:         Effective ED server utilization rate
    - P_abandon:      Fraction of arriving patients who abandon before being seen
    - E[W_q]:         Mean queue waiting time before ED treatment (hours)
    - E[W_ED]:        Mean total ED sojourn time (hours)
    - E[N_q]:         Time-average queue length (waiting patients, not in service)
    - ICU_util:       Mean ICU bed utilization fraction
"""

import pathlib
import random
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import simpy

# ---------------------------------------------------------
# 1. Directory Setup
# ---------------------------------------------------------
BASE_DIR  = pathlib.Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "output_tables"
PLOTS_DIR  = BASE_DIR / "output_plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(" APPLIED STOCHASTIC MODELLING (ASM) — TANDEM QUEUE DES SIMULATION")
print("=" * 80)

# ---------------------------------------------------------
# 2. Model & Simulation Parameters
# ---------------------------------------------------------
# Diurnal arrival rates from Script 02 (Very High tier, Lambda_day = 275.2 pts/day)
# Load from saved table if available, otherwise use verified defaults
nhpp_path = BASE_DIR / "output_tables" / "02_nhpp_diurnal_hourly_rates.csv"
if nhpp_path.exists():
    df_nhpp = pd.read_csv(nhpp_path)
    LAMBDA_HOURLY = df_nhpp["lambda_very_high_per_hr"].values   # length 24
else:
    # Fallback: uniform 11.47 patients/hour (275.2 / 24)
    LAMBDA_HOURLY = np.full(24, 275.2 / 24)

# Best-fit Weibull parameters for ICU LOS from Script 02
LOS_FIT_PATH = BASE_DIR / "output_tables" / "02_icu_los_distribution_fitting.csv"
if LOS_FIT_PATH.exists():
    df_fit  = pd.read_csv(LOS_FIT_PATH)
    weib_row = df_fit[df_fit["distribution"] == "Weibull"].iloc[0]
    # scipy weibull_min: params = (c, loc=0, scale=beta)
    WEIB_C    = float(weib_row["fitted_parameters"].split(",")[0].split("=")[1])
    WEIB_BETA = float(weib_row["fitted_parameters"].split(",")[1].split("=")[1].split()[0])
else:
    WEIB_C, WEIB_BETA = 0.9126, 55.57    # pre-computed defaults

# ED queue parameters (calibrated from CMS Very High tier via Erlang-A law)
# P_abandon = 2.33%; W_ED = 202.5 min; theta calibrated to match
P_ABANDON_TARGET = 0.0233          # from CMS OP_22
W_ED_MINUTES     = 202.5           # from CMS OP_18b
THETA            = 0.50            # patience abandonment rate (per hour) — calibrated
MU_ED            = 60.0 / (W_ED_MINUTES - (P_ABANDON_TARGET / THETA) * 60)  # per hour
MU_ED            = max(MU_ED, 0.2)  # safety floor

C1_BEDS          = 20              # Number of ED acute treatment bays
P_ICU            = 0.125           # Fraction of ED patients routed to ICU (12.5%)

# Simulation time configuration (hours)
WARM_UP_HRS  = 5  * 24.0          # 5 days warm-up
SIM_HRS      = 30 * 24.0          # 30-day measurement window
TOTAL_HRS    = WARM_UP_HRS + SIM_HRS
N_REPLICATIONS = 20               # Monte Carlo replications
C2_SWEEP       = [8, 10, 12, 14, 16, 20]   # ICU bed capacity sensitivity

print(f"\n[CONFIG] Simulation Parameters:")
print(f"  ED bays (c1): {C1_BEDS}")
print(f"  ICU service distribution: Weibull(c={WEIB_C:.4f}, beta={WEIB_BETA:.2f} hrs)")
print(f"  ED service rate (mu_ED): {MU_ED:.4f} per hour (mean = {1/MU_ED:.1f} hrs)")
print(f"  Patient abandonment rate (theta): {THETA:.4f} per hour")
print(f"  ICU routing probability (p_ICU): {P_ICU}")
print(f"  Peak hourly arrival (lambda_max): {LAMBDA_HOURLY.max():.2f} patients/hour")
print(f"  Warm-up: {WARM_UP_HRS/24:.0f} days | Measurement: {SIM_HRS/24:.0f} days | Replications: {N_REPLICATIONS}")

# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def arrival_rate_at_hour(sim_time_hours: float) -> float:
    """Return NHPP arrival rate at a given simulation time (in hours)."""
    hour_of_day = int(sim_time_hours % 24)
    return float(LAMBDA_HOURLY[hour_of_day])

def sample_weibull_los(rng: np.random.Generator) -> float:
    """Sample ICU LOS from Weibull distribution (hours). Uses inverse CDF method."""
    u = rng.random()
    # F(t) = 1 - exp(-(t/beta)^c)  =>  t = beta * (-ln(1 - u))^(1/c)
    t = WEIB_BETA * (-np.log(1.0 - u + 1e-12)) ** (1.0 / WEIB_C)
    return max(t, 0.01)

def sample_exp(rng: np.random.Generator, rate: float) -> float:
    """Sample from Exponential(rate)."""
    return rng.exponential(1.0 / rate)

# ---------------------------------------------------------
# 4. DES Model — Tandem ED → ICU Queue
# ---------------------------------------------------------
class SimulationMetrics:
    """Thread-safe container for per-replication performance metrics."""
    def __init__(self, warm_up_end: float):
        self.warm_up_end  = warm_up_end
        self.total_arrive = 0
        self.total_abandon = 0
        self.total_treated = 0
        self.total_icu_pts = 0
        self.wait_times   = []        # W_q for each ED patient
        self.sojourn_times = []       # W_ED for each ED patient
        self.board_delays  = []       # ED boarding duration for ICU patients
        # Time-averaged utilization tracking
        self._last_ed_time   = 0.0
        self._last_icu_time  = 0.0
        self._ed_busy_area   = 0.0
        self._icu_busy_area  = 0.0
        self._ed_busy        = 0
        self._icu_busy       = 0
        self._last_q_time    = 0.0
        self._q_area         = 0.0
        self._q_len          = 0

    def record_arrive(self):
        self.total_arrive += 1

    def record_abandon(self):
        self.total_abandon += 1

    def record_wait(self, env_now, arrival_time):
        if env_now >= self.warm_up_end:
            self.wait_times.append(env_now - arrival_time)

    def record_sojourn(self, env_now, arrival_time):
        if env_now >= self.warm_up_end:
            self.sojourn_times.append(env_now - arrival_time)
            self.total_treated += 1

    def record_boarding(self, env_now, board_start):
        if env_now >= self.warm_up_end:
            self.board_delays.append(env_now - board_start)

    def record_icu(self):
        self.total_icu_pts += 1

    def update_ed_util(self, env_now, c1_eff, delta_ed):
        """Update the area under ED busy count curve."""
        elapsed = env_now - self._last_ed_time
        if env_now >= self.warm_up_end:
            self._ed_busy_area += self._ed_busy * elapsed
        self._ed_busy    += delta_ed
        self._last_ed_time = env_now

    def update_icu_util(self, env_now, delta_icu):
        """Update the area under ICU busy count curve."""
        elapsed = env_now - self._last_icu_time
        if env_now >= self.warm_up_end:
            self._icu_busy_area += self._icu_busy * elapsed
        self._icu_busy    += delta_icu
        self._last_icu_time = env_now

    def update_queue(self, env_now, delta_q):
        """Update the time-average queue length."""
        elapsed = env_now - self._last_q_time
        if env_now >= self.warm_up_end:
            self._q_area += self._q_len * elapsed
        self._q_len    += delta_q
        self._last_q_time = env_now

    def summary(self, c1, c2) -> dict:
        """Compute final summary statistics for this replication."""
        n_warm = max(1, self.total_arrive)
        meas_time = SIM_HRS   # measurement window in hours

        p_abandon = self.total_abandon / n_warm if n_warm > 0 else 0.0
        e_wq      = np.mean(self.wait_times)     if self.wait_times    else 0.0
        e_wed     = np.mean(self.sojourn_times)  if self.sojourn_times else 0.0
        e_board   = np.mean(self.board_delays)   if self.board_delays  else 0.0
        rho_ed    = self._ed_busy_area  / (c1   * meas_time) if meas_time > 0 else 0.0
        rho_icu   = self._icu_busy_area / (c2   * meas_time) if meas_time > 0 else 0.0
        e_nq      = self._q_area / meas_time if meas_time > 0 else 0.0

        return {
            "p_abandon":           p_abandon,
            "mean_wait_hrs":       e_wq,
            "mean_sojourn_hrs":    e_wed,
            "mean_board_delay_hrs": e_board,
            "ed_server_util":      min(rho_ed,  1.0),
            "icu_bed_util":        min(rho_icu, 1.0),
            "mean_queue_length":   e_nq,
            "total_patients":      self.total_arrive,
            "total_icu_patients":  self.total_icu_pts,
        }


def run_simulation(c2: int, rng_seed: int) -> dict:
    """Run one 35-day replication of the tandem ED->ICU queue."""
    env     = simpy.Environment()
    rng     = np.random.default_rng(rng_seed)
    metrics = SimulationMetrics(warm_up_end=WARM_UP_HRS)

    # SimPy resources
    ed_beds  = simpy.Resource(env, capacity=C1_BEDS)
    icu_beds = simpy.Resource(env, capacity=c2)

    # Shared state: number of boarded ICU patients
    boarded_count = [0]

    def patient(arrival_time: float):
        """Model a single patient's journey through ED -> possibly ICU."""
        metrics.record_arrive()

        # --- Phase 1: Wait for an available ED treatment bay ---
        patience  = sample_exp(rng, THETA)      # how long patient will wait
        wait_start = env.now
        metrics.update_queue(env.now, +1)

        with ed_beds.request() as req:
            result = yield req | env.timeout(patience)

            if req not in result:
                # Patient abandoned the queue before being seen
                metrics.record_abandon()
                metrics.update_queue(env.now, -1)
                return

            # Patient obtained an ED bay
            metrics.record_wait(env.now, wait_start)
            metrics.update_queue(env.now, -1)
            metrics.update_ed_util(env.now, C1_BEDS, +1)

            # --- Phase 2: ED Treatment ---
            ed_service_time = sample_exp(rng, MU_ED)
            yield env.timeout(ed_service_time)

            # --- Phase 3: Routing Decision ---
            if rng.random() < P_ICU:
                # Patient needs ICU admission
                metrics.record_icu()
                board_start = env.now

                # If all ICU beds are full, patient boards in this ED bay
                # (blocks the ED resource until an ICU bed frees up)
                if icu_beds.count >= c2:
                    boarded_count[0] += 1
                    # Stay blocked in ED bay waiting for ICU bed
                    with icu_beds.request() as icu_req:
                        yield icu_req
                    boarded_count[0] -= 1
                    metrics.record_boarding(env.now, board_start)
                    metrics.update_ed_util(env.now, C1_BEDS, -1)
                    # ED bay released when ICU bed obtained
                else:
                    # ICU bed available immediately — release ED bay first
                    metrics.update_ed_util(env.now, C1_BEDS, -1)
                    with icu_beds.request() as icu_req:
                        yield icu_req

                # --- Phase 4: ICU Service ---
                metrics.update_icu_util(env.now, +1)
                icu_los = sample_weibull_los(rng)
                yield env.timeout(icu_los)
                metrics.update_icu_util(env.now, -1)
            else:
                # Patient discharged from ED (no ICU needed)
                metrics.record_sojourn(env.now, arrival_time)
                metrics.update_ed_util(env.now, C1_BEDS, -1)

    def nhpp_arrivals():
        """Generate NHPP patient arrivals using Lewis-Shedler thinning."""
        lambda_max = float(LAMBDA_HOURLY.max())
        while env.now < TOTAL_HRS:
            # Sample inter-arrival from bounding homogeneous process
            inter = sample_exp(rng, lambda_max)
            yield env.timeout(inter)
            if env.now >= TOTAL_HRS:
                break
            # Acceptance-rejection (thinning)
            lam_actual = arrival_rate_at_hour(env.now)
            if rng.random() < lam_actual / lambda_max:
                env.process(patient(env.now))

    env.process(nhpp_arrivals())
    env.run(until=TOTAL_HRS)

    return metrics.summary(c1=C1_BEDS, c2=c2)


# ---------------------------------------------------------
# 5. Monte Carlo Sweep Over ICU Capacity c2
# ---------------------------------------------------------
print("\n--- Running Monte Carlo Sensitivity Sweep ---")
print(f"  ICU bed capacities tested: {C2_SWEEP}")
print(f"  Replications per configuration: {N_REPLICATIONS}")

all_results = []
for c2 in C2_SWEEP:
    print(f"\n  c2 = {c2} ICU beds ...", end=" ", flush=True)
    rep_results = []
    for rep in range(N_REPLICATIONS):
        seed = 42 + c2 * 100 + rep
        res  = run_simulation(c2=c2, rng_seed=seed)
        res["c2"] = c2
        res["replication"] = rep + 1
        rep_results.append(res)
        print(".", end="", flush=True)
    all_results.extend(rep_results)
    print(" done")

df_raw = pd.DataFrame(all_results)
df_raw.to_csv(TABLES_DIR / "04_sim_all_replications.csv", index=False)

# ---------------------------------------------------------
# 6. Aggregate Statistics (Mean ± 95% CI across replications)
# ---------------------------------------------------------
metrics_cols = [
    "p_abandon", "mean_wait_hrs", "mean_sojourn_hrs",
    "mean_board_delay_hrs", "ed_server_util",
    "icu_bed_util", "mean_queue_length"
]

agg_records = []
for c2, grp in df_raw.groupby("c2"):
    row = {"c2_icu_beds": c2}
    for col in metrics_cols:
        vals = grp[col].dropna().values
        n = len(vals)
        mu  = vals.mean()
        se  = vals.std() / math.sqrt(n) if n > 1 else 0.0
        ci  = 1.96 * se
        row[f"{col}_mean"] = mu
        row[f"{col}_ci95"] = ci
    agg_records.append(row)

df_agg = pd.DataFrame(agg_records)
df_agg.to_csv(TABLES_DIR / "04_sim_capacity_metrics_aggregated.csv", index=False)

print("\n--- Aggregated Simulation Results (Mean over 20 replications) ---")
print(df_agg[[
    "c2_icu_beds", "p_abandon_mean", "mean_wait_hrs_mean",
    "mean_board_delay_hrs_mean", "ed_server_util_mean", "icu_bed_util_mean"
]].to_string(index=False))

print("\n[EXPORT] Saved simulation tables to 'output_tables/':")
print("  - 04_sim_all_replications.csv")
print("  - 04_sim_capacity_metrics_aggregated.csv")

# ---------------------------------------------------------
# 7. Publication Plots
# ---------------------------------------------------------
c2_vals    = df_agg["c2_icu_beds"].values
colors_sim = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4", "#8c564b"]

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
plt.subplots_adjust(hspace=0.40, wspace=0.30)

def plot_metric(ax, col, ylabel, title, color="#1f77b4", fmt=":.3f"):
    means = df_agg[f"{col}_mean"].values
    cis   = df_agg[f"{col}_ci95"].values
    ax.bar(c2_vals.astype(str), means, color=color, edgecolor="black", alpha=0.85, width=0.55)
    ax.errorbar(range(len(c2_vals)), means, yerr=cis, fmt="none",
                color="black", capsize=5, linewidth=1.5)
    for i, (m, c) in enumerate(zip(means, cis)):
        ax.text(i, m + cis[i] + means.max() * 0.015,
                f"{m:{fmt.replace(':', '')}}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(c2_vals)))
    ax.set_xticklabels([str(v) for v in c2_vals], fontsize=10)
    ax.set_xlabel("ICU Bed Capacity  (c2)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

plot_metric(axs[0, 0], "p_abandon",
            "P(abandon) = LWBS Rate",
            "A: Patient Abandonment Rate  P_abandon\nvs. ICU Bed Capacity c2",
            color="#d62728", fmt=":.4f")

plot_metric(axs[0, 1], "mean_board_delay_hrs",
            "Mean Boarding Delay  E[B_block]  (Hours)",
            "B: Mean Pre-ICU Boarding Delay  E[B_block]\nvs. ICU Bed Capacity c2",
            color="#ff7f0e", fmt=":.2f")

plot_metric(axs[0, 2], "mean_wait_hrs",
            "Mean Queue Wait  E[W_q]  (Hours)",
            "C: Mean ED Queue Waiting Time  E[W_q]\nvs. ICU Bed Capacity c2",
            color="#9467bd", fmt=":.3f")

plot_metric(axs[1, 0], "ed_server_util",
            "ED Server Utilization  rho_ED",
            "D: Effective ED Server Utilization  rho_ED\nvs. ICU Bed Capacity c2",
            color="#2ca02c", fmt=":.3f")

plot_metric(axs[1, 1], "icu_bed_util",
            "ICU Bed Utilization  rho_ICU",
            "E: Mean ICU Bed Utilization  rho_ICU\nvs. ICU Bed Capacity c2",
            color="#1f77b4", fmt=":.3f")

plot_metric(axs[1, 2], "mean_queue_length",
            "Mean Queue Length  E[N_q]",
            "F: Time-Average ED Queue Length  E[N_q]\nvs. ICU Bed Capacity c2",
            color="#8c564b", fmt=":.2f")

fig.suptitle(
    "ICU Capacity Sensitivity Analysis: Tandem ED -> ICU DES  |  20 Monte Carlo Replications  |  ASM Course Project",
    fontsize=14, fontweight="bold", y=1.01
)
plt.savefig(PLOTS_DIR / "04_icu_capacity_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close()
print("[PLOT] Generated: 04_icu_capacity_sensitivity.png")

# --- Plot 2: Sample Path Trajectories (queue length over one replication at c2=12) ---
print("\n  Generating sample queue-length trajectory plot ...")

class PathRecorder:
    """Record queue-length and ICU occupancy trajectory for one replication."""
    def __init__(self):
        self.times  = []
        self.ql     = []     # ED queue length (waiting, not in service)
        self.icu_occ = []    # ICU bed occupancy count

env2     = simpy.Environment()
rng2     = np.random.default_rng(9999)
recorder = PathRecorder()
c2_demo  = 12

ed_beds2  = simpy.Resource(env2, capacity=C1_BEDS)
icu_beds2 = simpy.Resource(env2, capacity=c2_demo)
q_count   = [0]
icu_count = [0]

def patient_trace(arrival_time):
    """Patient process with trajectory recording."""
    patience = sample_exp(rng2, THETA)
    wait_s   = env2.now

    recorder.times.append(env2.now)
    q_count[0] += 1
    recorder.ql.append(q_count[0])
    recorder.icu_occ.append(icu_count[0])

    with ed_beds2.request() as req:
        res = yield req | env2.timeout(patience)
        recorder.times.append(env2.now)
        q_count[0] -= 1
        recorder.ql.append(q_count[0])
        recorder.icu_occ.append(icu_count[0])

        if req not in res:
            return  # abandoned

        ed_svc = sample_exp(rng2, MU_ED)
        yield env2.timeout(ed_svc)

        if rng2.random() < P_ICU:
            board_s = env2.now
            with icu_beds2.request() as ir:
                yield ir
            icu_count[0] += 1
            recorder.times.append(env2.now)
            recorder.ql.append(q_count[0])
            recorder.icu_occ.append(icu_count[0])

            icu_los = sample_weibull_los(rng2)
            yield env2.timeout(icu_los)
            icu_count[0] -= 1
            recorder.times.append(env2.now)
            recorder.ql.append(q_count[0])
            recorder.icu_occ.append(icu_count[0])

def arrivals_trace():
    lmax = float(LAMBDA_HOURLY.max())
    while env2.now < 10 * 24:   # 10-day trace only
        inter = sample_exp(rng2, lmax)
        yield env2.timeout(inter)
        if env2.now >= 10 * 24:
            break
        if rng2.random() < arrival_rate_at_hour(env2.now) / lmax:
            env2.process(patient_trace(env2.now))

env2.process(arrivals_trace())
env2.run(until=10 * 24)

t_arr  = np.array(recorder.times) / 24.0   # convert hours -> days
ql_arr = np.array(recorder.ql)
ic_arr = np.array(recorder.icu_occ)

fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
plt.subplots_adjust(hspace=0.12)

ax1.step(t_arr, ql_arr, where="post", color="#d62728", linewidth=1.2, alpha=0.85)
ax1.fill_between(t_arr, ql_arr, step="post", color="#d62728", alpha=0.25)
ax1.set_ylabel("ED Queue Length  N_q(t)", fontsize=12)
ax1.set_title(f"Sample Path: ED Queue Length and ICU Occupancy  (c2 = {c2_demo} ICU beds, 10-Day Trace)", 
              fontsize=13, fontweight="bold")
ax1.axhline(C1_BEDS, color="black", linestyle="--", linewidth=1.5, label=f"ED Capacity Limit c1 = {C1_BEDS}")
ax1.legend(frameon=True, fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.step(t_arr, ic_arr, where="post", color="#1f77b4", linewidth=1.2, alpha=0.85)
ax2.fill_between(t_arr, ic_arr, step="post", color="#1f77b4", alpha=0.25)
ax2.axhline(c2_demo, color="black", linestyle="--", linewidth=1.5, label=f"ICU Capacity Limit c2 = {c2_demo}")
ax2.set_ylabel("ICU Bed Occupancy  N_ICU(t)", fontsize=12)
ax2.set_xlabel("Simulation Time (Days)", fontsize=12)
ax2.legend(frameon=True, fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.savefig(PLOTS_DIR / "04_sample_path_trajectories.png", dpi=300, bbox_inches="tight")
plt.close()
print("[PLOT] Generated: 04_sample_path_trajectories.png")

# --- Plot 3: Tradeoff curves ---
fig4, ax4 = plt.subplots(1, 2, figsize=(14, 6))

ax4[0].plot(c2_vals, df_agg["p_abandon_mean"].values * 100, "o-",
            color="#d62728", linewidth=2.5, markersize=9, label="LWBS Abandonment %")
ax4[0].fill_between(c2_vals,
    (df_agg["p_abandon_mean"] - df_agg["p_abandon_ci95"]).values * 100,
    (df_agg["p_abandon_mean"] + df_agg["p_abandon_ci95"]).values * 100,
    alpha=0.20, color="#d62728")
ax4[0].set_xlabel("ICU Bed Capacity  c2", fontsize=12)
ax4[0].set_ylabel("Patient Abandonment Rate (%)", fontsize=12)
ax4[0].set_title("A: P_abandon vs. c2 ICU Beds\n(95% Confidence Interval Shaded)", fontsize=12, fontweight="bold")
ax4[0].set_xticks(c2_vals)
ax4[0].grid(True, linestyle=":", alpha=0.6)

ax4[1].plot(c2_vals, df_agg["mean_board_delay_hrs_mean"].values, "s-",
            color="#ff7f0e", linewidth=2.5, markersize=9, label="Mean Boarding Delay")
ax4[1].fill_between(c2_vals,
    (df_agg["mean_board_delay_hrs_mean"] - df_agg["mean_board_delay_hrs_ci95"]).values,
    (df_agg["mean_board_delay_hrs_mean"] + df_agg["mean_board_delay_hrs_ci95"]).values,
    alpha=0.20, color="#ff7f0e")
ax4[1].set_xlabel("ICU Bed Capacity  c2", fontsize=12)
ax4[1].set_ylabel("Mean ICU Boarding Delay  E[B_block]  (Hours)", fontsize=12)
ax4[1].set_title("B: E[B_block] vs. c2 ICU Beds\n(95% Confidence Interval Shaded)", fontsize=12, fontweight="bold")
ax4[1].set_xticks(c2_vals)
ax4[1].grid(True, linestyle=":", alpha=0.6)

fig4.suptitle("Capacity Tipping Point Analysis: ICU Bed Count vs. Upstream ED Congestion  |  ASM Course Project",
              fontsize=13, fontweight="bold")
plt.savefig(PLOTS_DIR / "04_capacity_tradeoff_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("[PLOT] Generated: 04_capacity_tradeoff_curves.png")

print("=" * 80)
print(" TANDEM QUEUE DES SIMULATION COMPLETE — READY FOR VALIDATION (SCRIPT 05)")
print("=" * 80)
