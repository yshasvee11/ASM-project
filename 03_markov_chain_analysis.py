#!/usr/bin/env python3
"""
03_markov_chain_analysis.py
Applied Stochastic Modelling (ASM) Course Project
Hybrid ED-to-ICU Queueing & Markov Network

Description:
  Constructs the multi-ward patient care trajectory as an Absorbing Markov Chain.

  State Space:
    S = { 0:ED,  1:ICU,  2:SDU,  3:Floor,  4:Home(Absorbing),  5:Expired(Absorbing) }

  Steps:
    1. Estimate the empirical one-step DTMC transition probability matrix P from
       eicu_demo_data/patient.csv using patienthealthsystemstayid linkage.
    2. Decompose P into canonical absorbing form:
         P = [ Q_T  | R  ]
             [  0   | I  ]
    3. Compute Fundamental Matrix:  N = (I - Q_T)^(-1)
    4. Compute Absorption Matrix:   B = N * R
    5. Build CTMC infinitesimal generator Q from P and empirical mean holding times.
    6. Compute transient state occupancy curves: P(t) = exp(Q * t)
    7. Generate all required plots and tables.
"""

import pathlib
import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------
# 1. Directory Setup
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "output_tables"
PLOTS_DIR  = BASE_DIR / "output_plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(" APPLIED STOCHASTIC MODELLING (ASM) — MARKOV CHAIN ROUTING ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------
# 2. Load & Normalize Patient Data
# ---------------------------------------------------------
patient_path = BASE_DIR / "eicu_demo_data" / "patient.csv"
df = pd.read_csv(patient_path, low_memory=False)
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# ---------------------------------------------------------
# 3. State Space & Ward Mapping
# ---------------------------------------------------------
# We map each patient encounter to one of 6 clinical states.
# For absorbing states: Home and Expired derive from unitdischargelocation / unitdischargestatus.

STATES       = ["ED", "ICU", "SDU", "Floor", "Home", "Expired"]
TRANSIENT    = ["ED", "ICU", "SDU", "Floor"]       # indices 0-3
ABSORBING    = ["Home", "Expired"]                  # indices 4-5
N_STATES     = len(STATES)
N_TRANSIENT  = len(TRANSIENT)
N_ABSORBING  = len(ABSORBING)
STATE_IDX    = {s: i for i, s in enumerate(STATES)}

def classify_unit_type(row):
    """Map the ICU unit type / admit source to a Markov state."""
    src = str(row.get("unitadmitsource", "")).strip()
    utype = str(row.get("unittype", "")).strip()
    stay = str(row.get("unitstaytype", "")).strip().lower()

    if "Emergency" in src:
        return "ED"
    elif "Step-Down" in src or "SDU" in src or "stepdown" in stay:
        return "SDU"
    elif "Floor" in src or "Acute Care" in src:
        return "Floor"
    else:
        return "ICU"

def classify_discharge_state(row):
    """Map unit discharge to a state (transient or absorbing)."""
    loc    = str(row.get("unitdischargelocation", "")).strip()
    status = str(row.get("unitdischargestatus", "")).strip()

    if status == "Expired" or "Death" in loc:
        return "Expired"
    elif "Home" in loc:
        return "Home"
    elif "Step-Down" in loc or "SDU" in loc or "Telemetry" in loc:
        return "SDU"
    elif "Floor" in loc or "Acute Care" in loc or "Nursing Home" in loc \
         or "Skilled Nursing" in loc or "Rehabilitation" in loc:
        return "Floor"
    elif "ICU" in loc or "Other ICU" in loc:
        return "ICU"
    elif "Other Hospital" in loc or "Other External" in loc:
        return "Home"    # treated as live discharge from this facility
    else:
        return "Floor"   # default to general ward

# ---------------------------------------------------------
# 4. Build Empirical Transition Matrix (DTMC)
# ---------------------------------------------------------
# For each patient hospitalization (linked by patienthealthsystemstayid),
# sort ICU stay records chronologically by unitvisitnumber and construct
# transitions: (from_state -> to_state) for each consecutive pair.
# The final discharge state (Home or Expired) is derived from hospitaldischargestatus.

print("\n--- Part 1: Constructing Empirical One-Step DTMC Transition Matrix ---")

df["ward_state"]     = df.apply(classify_unit_type, axis=1)
df["discharge_state"] = df.apply(classify_discharge_state, axis=1)

# Sort all stays by hospitalization then unit visit number
df_sorted = df.sort_values(["patienthealthsystemstayid", "unitvisitnumber"])

# Count raw transition frequencies
transition_counts = np.zeros((N_STATES, N_STATES), dtype=float)

# Build transitions within each hospital stay
for stay_id, group in df_sorted.groupby("patienthealthsystemstayid"):
    rows = group.reset_index(drop=True)
    n    = len(rows)

    for i in range(n):
        from_state = rows.loc[i, "ward_state"]

        if i < n - 1:
            # Mid-stay transition: go to next ICU unit
            to_state = rows.loc[i + 1, "ward_state"]
        else:
            # Final discharge for this stay
            to_state = rows.loc[i, "discharge_state"]

        # Map final hospital discharge status: override terminal state
        if i == n - 1:
            hosp_status = str(rows.loc[i, "hospitaldischargestatus"]).strip()
            hosp_loc    = str(rows.loc[i, "hospitaldischargelocation"]).strip()
            if hosp_status == "Expired":
                to_state = "Expired"
            elif "Home" in hosp_loc or "Home" == to_state:
                to_state = "Home"

        fi = STATE_IDX[from_state]
        ti = STATE_IDX.get(to_state, STATE_IDX["Floor"])
        transition_counts[fi, ti] += 1.0

# Add small smoothing to avoid zero rows in transient states
for i in range(N_TRANSIENT):
    if transition_counts[i].sum() == 0:
        transition_counts[i, STATE_IDX["Floor"]] = 1.0

# Normalize rows to form probability matrix
P = np.zeros((N_STATES, N_STATES))
for i in range(N_STATES):
    row_sum = transition_counts[i].sum()
    if row_sum > 0:
        P[i] = transition_counts[i] / row_sum
    else:
        P[i, i] = 1.0   # stay in absorbing state

# Absorbing states: ensure P[Home, Home] = 1 and P[Expired, Expired] = 1
P[STATE_IDX["Home"], :]    = 0.0
P[STATE_IDX["Home"], STATE_IDX["Home"]] = 1.0
P[STATE_IDX["Expired"], :] = 0.0
P[STATE_IDX["Expired"], STATE_IDX["Expired"]] = 1.0

df_P = pd.DataFrame(P, index=STATES, columns=STATES)
print("\nOne-Step Transition Probability Matrix P:")
print(df_P.round(4).to_string())

# ---------------------------------------------------------
# 5. Canonical Decomposition: Q_T, R, and Fundamental Matrix N
# ---------------------------------------------------------
print("\n--- Part 2: Canonical Decomposition & Fundamental Matrix N = (I - Q_T)^(-1) ---")

# Reorder: transient first, absorbing last
order  = TRANSIENT + ABSORBING
idx_T  = [STATE_IDX[s] for s in TRANSIENT]
idx_A  = [STATE_IDX[s] for s in ABSORBING]

Q_T = P[np.ix_(idx_T, idx_T)]   # 4x4 sub-matrix of transient-to-transient
R   = P[np.ix_(idx_T, idx_A)]   # 4x2 transient-to-absorbing

I_T = np.eye(N_TRANSIENT)
N_fund = np.linalg.inv(I_T - Q_T)   # Fundamental matrix

# Expected visits to each transient state before absorption
expected_visits = pd.DataFrame(N_fund, index=TRANSIENT, columns=TRANSIENT)

# Expected total stages before discharge/death
t_absorb = N_fund @ np.ones(N_TRANSIENT)
expected_steps_df = pd.DataFrame({"expected_steps_to_absorption": t_absorb}, index=TRANSIENT)

# Absorption probability matrix B = N * R
B = N_fund @ R
B_df = pd.DataFrame(B, index=TRANSIENT, columns=ABSORBING)

print("\nFundamental Matrix N = (I - Q_T)^(-1):")
print(expected_visits.round(4).to_string())

print("\nExpected Steps Before Absorption (t = N * 1):")
print(expected_steps_df.round(4).to_string())

print("\nAbsorption Probability Matrix B = N * R:")
print(B_df.round(4).to_string())

# ---------------------------------------------------------
# 6. CTMC Generator Matrix Q
# ---------------------------------------------------------
print("\n--- Part 3: Constructing CTMC Infinitesimal Generator Q ---")

# Compute mean ICU LOS per source state (hours)
# Use unitdischargeoffset for ICU stays, approximate ED hold times from boarding data
los_by_state = {}

icu_los_hr = df["unitdischargeoffset"].dropna() / 60.0
mean_icu_los = icu_los_hr[icu_los_hr > 0].mean()

# Approximate mean holding times per ward (hours) from data
los_by_state["ED"]    = df[df["ward_state"] == "ED"]["unitdischargeoffset"].dropna().mean() / 60.0
los_by_state["ICU"]   = mean_icu_los
los_by_state["SDU"]   = df[df["ward_state"] == "SDU"]["unitdischargeoffset"].dropna().mean() / 60.0
los_by_state["Floor"] = df[df["ward_state"] == "Floor"]["unitdischargeoffset"].dropna().mean() / 60.0

# Fallback defaults if insufficient data
defaults = {"ED": 4.5, "ICU": 58.07, "SDU": 36.0, "Floor": 72.0}
for state in TRANSIENT:
    if np.isnan(los_by_state.get(state, np.nan)) or los_by_state[state] <= 0:
        los_by_state[state] = defaults[state]

mean_los_df = pd.DataFrame.from_dict(
    {s: {"mean_holding_time_hours": los_by_state[s], "rate_q_ii": -1.0 / los_by_state[s]}
     for s in TRANSIENT}, orient="index"
)
print("\nMean Holding Times per Ward:")
print(mean_los_df.round(4).to_string())

# Build generator matrix Q (6x6 for all states)
Q_gen = np.zeros((N_STATES, N_STATES))

for i_name in TRANSIENT:
    i = STATE_IDX[i_name]
    q_ii = -1.0 / los_by_state[i_name]  # diagonal = -1 / E[S_i]
    Q_gen[i, i] = q_ii
    for j_name in STATES:
        j = STATE_IDX[j_name]
        if j != i:
            Q_gen[i, j] = -q_ii * P[i, j]   # q_ij = -q_ii * P_ij

# Absorbing states: zero rows
Q_gen[STATE_IDX["Home"], :]    = 0.0
Q_gen[STATE_IDX["Expired"], :] = 0.0

df_Q = pd.DataFrame(Q_gen, index=STATES, columns=STATES)
print("\nCTMC Infinitesimal Generator Q:")
print(df_Q.round(6).to_string())

# ---------------------------------------------------------
# 7. Transient State Occupancy Curves: P(t) = exp(Q * t)
# ---------------------------------------------------------
print("\n--- Part 4: Computing Transient State Trajectories P(t) = exp(Q * t) ---")

# Evaluate over t = 0 to 14 days (in hours)
t_days_range = np.linspace(0, 14, 300)
t_hours      = t_days_range * 24.0

# Starting from ED (most common entry point)
init_state = np.zeros(N_STATES)
init_state[STATE_IDX["ED"]] = 1.0

P_t_all = []
for t in t_hours:
    exp_Qt = scipy.linalg.expm(Q_gen * t)
    P_t_all.append(init_state @ exp_Qt)

P_t_array = np.array(P_t_all)   # shape (300, 6)

print(f"  State probabilities computed for {len(t_hours)} time points over 0-14 days.")
print(f"  Steady-state (t=14 days): {dict(zip(STATES, P_t_array[-1].round(4)))}")

# ---------------------------------------------------------
# 8. Export Tables
# ---------------------------------------------------------
df_P.round(6).to_csv(TABLES_DIR / "03_dtmc_transition_matrix_P.csv")
B_df.round(6).to_csv(TABLES_DIR / "03_absorption_probabilities_B.csv")
expected_visits.round(6).to_csv(TABLES_DIR / "03_fundamental_matrix_N.csv")
expected_steps_df.round(4).to_csv(TABLES_DIR / "03_expected_steps_to_absorption.csv")
df_Q.round(6).to_csv(TABLES_DIR / "03_ctmc_generator_Q.csv")

traj_df = pd.DataFrame(P_t_array, columns=STATES)
traj_df.insert(0, "time_hours", t_hours)
traj_df.insert(1, "time_days", t_days_range)
traj_df.to_csv(TABLES_DIR / "03_ctmc_transient_trajectories.csv", index=False)

print("\n[EXPORT] Saved Markov Chain tables to 'output_tables/':")
print("  - 03_dtmc_transition_matrix_P.csv")
print("  - 03_absorption_probabilities_B.csv")
print("  - 03_fundamental_matrix_N.csv")
print("  - 03_expected_steps_to_absorption.csv")
print("  - 03_ctmc_generator_Q.csv")
print("  - 03_ctmc_transient_trajectories.csv")

# ---------------------------------------------------------
# 9. Publication Plots
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(16, 13))
plt.subplots_adjust(hspace=0.38, wspace=0.28)

# --- Plot A: DTMC Transition Matrix Heatmap ---
ax = axs[0, 0]
P_display = df_P.copy()
mask = P_display == 0
cmap = plt.cm.Blues
im = ax.imshow(P_display.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Transition Probability")
ax.set_xticks(range(N_STATES))
ax.set_yticks(range(N_STATES))
ax.set_xticklabels(STATES, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(STATES, fontsize=10)
for i in range(N_STATES):
    for j in range(N_STATES):
        val = P_display.iloc[i, j]
        if val > 0.001:
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)
ax.set_title("A: One-Step DTMC Transition Probability Matrix P\n(From Ward State i to Ward State j)", 
             fontsize=12, fontweight="bold")
ax.set_xlabel("To State (j)", fontsize=11)
ax.set_ylabel("From State (i)", fontsize=11)

# --- Plot B: Absorption Probabilities by Entry Ward ---
ax = axs[0, 1]
x  = np.arange(N_TRANSIENT)
w  = 0.35
bars1 = ax.bar(x - w/2, B_df["Home"].values,    width=w, color="#2ca02c", edgecolor="black", label="Discharged Alive (Home)")
bars2 = ax.bar(x + w/2, B_df["Expired"].values, width=w, color="#d62728", edgecolor="black", label="In-Hospital Mortality (Expired)")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(TRANSIENT, fontsize=11)
ax.set_ylabel("Absorption Probability  B = N * R", fontsize=11)
ax.set_title("B: Absorption Probability Matrix  B = N * R\n(Survival vs. Mortality by Entry Ward)", 
             fontsize=12, fontweight="bold")
ax.legend(frameon=True, fontsize=10)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# --- Plot C: Transient State Probability Trajectories (0 to 14 days) ---
ax = axs[1, 0]
colors_traj = {"ED": "#1f77b4", "ICU": "#ff7f0e", "SDU": "#9467bd",
               "Floor": "#2ca02c", "Home": "#17becf", "Expired": "#d62728"}
for state in STATES:
    idx = STATE_IDX[state]
    ls  = "-" if state in TRANSIENT else "--"
    lw  = 2.2 if state in TRANSIENT else 1.8
    ax.plot(t_days_range, P_t_array[:, idx], label=state, color=colors_traj[state], 
            linewidth=lw, linestyle=ls)
ax.set_xlabel("Time (Days from ED Admission)", fontsize=11)
ax.set_ylabel("State Occupancy Probability  P(t)", fontsize=11)
ax.set_title("C: CTMC Transient State Trajectories  P(t) = exp(Q * t)\n(Starting State: Emergency Department)", 
             fontsize=12, fontweight="bold")
ax.legend(frameon=True, fontsize=10, loc="center right")
ax.set_xlim(0, 14)
ax.set_ylim(-0.02, 1.05)
ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
ax.grid(True, linestyle=":", alpha=0.6)

# --- Plot D: Fundamental Matrix N Heatmap ---
ax = axs[1, 1]
im2 = ax.imshow(N_fund, cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax, label="Expected Number of Visits")
ax.set_xticks(range(N_TRANSIENT))
ax.set_yticks(range(N_TRANSIENT))
ax.set_xticklabels(TRANSIENT, fontsize=10)
ax.set_yticklabels(TRANSIENT, fontsize=10)
for i in range(N_TRANSIENT):
    for j in range(N_TRANSIENT):
        ax.text(j, i, f"{N_fund[i, j]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_title("D: Fundamental Matrix  N = (I - Q_T)^(-1)\n(Expected Visits to Ward j Starting from Ward i)", 
             fontsize=12, fontweight="bold")
ax.set_xlabel("Target Ward (j)", fontsize=11)
ax.set_ylabel("Entry Ward (i)", fontsize=11)

fig.suptitle("Multi-Ward Patient Care Trajectory Analysis: Absorbing DTMC & CTMC  |  ASM Course Project",
             fontsize=14, fontweight="bold", y=1.00)
plt.savefig(PLOTS_DIR / "03_markov_chain_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("[PLOT] Generated: 03_markov_chain_analysis.png")

# --- Extra Plot: CTMC Transient states only (zoomed 0-5 days) ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
for state in TRANSIENT:
    idx = STATE_IDX[state]
    ax2.plot(t_days_range, P_t_array[:, idx], label=state, color=colors_traj[state], linewidth=2.5)
for state in ABSORBING:
    idx = STATE_IDX[state]
    ax2.plot(t_days_range, P_t_array[:, idx], label=state, color=colors_traj[state], 
             linewidth=2.0, linestyle="--", alpha=0.85)
ax2.set_xlabel("Time (Days from ED Admission)", fontsize=12)
ax2.set_ylabel("State Occupancy Probability  P(t)", fontsize=12)
ax2.set_title("CTMC State Probability Trajectories: P(t) = exp(Q * t)  |  Starting State: ED",
              fontsize=13, fontweight="bold")
ax2.legend(frameon=True, fontsize=11, loc="center right")
ax2.set_xlim(0, 14)
ax2.set_ylim(-0.02, 1.05)
ax2.grid(True, linestyle=":", alpha=0.6)
plt.savefig(PLOTS_DIR / "03_ctmc_state_trajectories.png", dpi=300, bbox_inches="tight")
plt.close()
print("[PLOT] Generated: 03_ctmc_state_trajectories.png")

print("=" * 80)
print(" MARKOV CHAIN ANALYSIS COMPLETE — READY FOR DES SIMULATION (SCRIPT 04)")
print("=" * 80)
