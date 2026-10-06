import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURATION & DIRECTORIES
# --------------------------------------------------
DATASET_PATH = os.path.join("dataset_v2_1", "wireless_queue_dataset_v2_1.csv")
FIGURES_DIR = os.path.join("dataset_v2_1", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE DATASET V2.1 SCIENTIFIC AUDIT")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset Shape: {df.shape}")

# --------------------------------------------------
# 1. CORE DISTRIBUTIONAL STATISTICS
# --------------------------------------------------
columns_to_check = [
    "queue_after",
    "arrival_rate",
    "interference",
    "service_capacity",
    "utilization",
    "estimated_delay",
    "packet_drops",
    "future_queue"
]

stats = df[columns_to_check].describe().T[["mean", "std", "min", "50%", "max"]]
stats.rename(columns={"50%": "median"}, inplace=True)
print("\n--- Summary Statistics ---")
print(stats.round(3))
stats.to_csv(os.path.join("dataset_v2_1", "descriptive_statistics.csv"))

# --------------------------------------------------
# 2. CONGESTION & TARGET DIFFERENTIAL ANALYSIS
# --------------------------------------------------
# Delta Q: transient change in queue length over single timestep
df["delta_queue"] = df["future_queue"] - df["queue_after"]

congested_mask = df["utilization"] > 1.0
normal_df = df[~congested_mask]
cong_df = df[congested_mask]

print("\n--- Operating Regime Breakdown ---")
print(f"Total Observations       : {len(df):,}")
print(f"Normal Samples (ρ <= 1)  : {len(normal_df):,} ({100 * len(normal_df)/len(df):.2f}%)")
print(f"Congested Samples (ρ > 1): {len(cong_df):,} ({100 * len(cong_df)/len(df):.2f}%)")
print(f"Samples with Drops (> 0) : {(df['packet_drops'] > 0).sum():,} ({100 * (df['packet_drops'] > 0).sum()/len(df):.2f}%)")

regime_comp = pd.DataFrame({
    "Normal Regime": normal_df[columns_to_check + ["delta_queue"]].mean(),
    "Congested Regime": cong_df[columns_to_check + ["delta_queue"]].mean()
})
print("\nRegime Mean Comparisons:")
print(regime_comp.round(3))
regime_comp.to_csv(os.path.join("dataset_v2_1", "regime_comparison.csv"))

# --------------------------------------------------
# 3. INTERFERENCE & CAPACITY COUPLING
# --------------------------------------------------
print("\n--- Interference Distribution & Empirical Service Capacity ---")
interf_profile = df.groupby("interference").agg(
    count=("service_capacity", "count"),
    mean_capacity=("service_capacity", "mean"),
    std_capacity=("service_capacity", "std")
).reset_index()
print(interf_profile.round(3))
interf_profile.to_csv(os.path.join("dataset_v2_1", "interference_profile.csv"), index=False)

# --------------------------------------------------
# 4. PUBLICATION FIGURES GENERATION
# --------------------------------------------------
print("\nGenerating figures...")

# Figure 1: Clean Single-Scenario Temporal Trajectory
plt.figure(figsize=(10, 5), dpi=300)
# Select Scenario 0 (or any representative scenario) to avoid cross-scenario interleaving
sample_scenario = df[df["scenario"] == 0]
for cid in range(min(5, sample_scenario["cell_id"].nunique())):
    cell_trace = sample_scenario[sample_scenario["cell_id"] == cid].sort_values("time")
    plt.plot(cell_trace["time"], cell_trace["queue_after"], label=f"Cell {cid}", linewidth=1.2)

plt.xlabel("Time Step (t)", fontsize=11)
plt.ylabel("Queue Length Q(t)", fontsize=11)
plt.title("Scenario 0: Queue Evolution Across Cells (Unmixed Time Horizon)", fontsize=12)
plt.legend(loc="upper right", frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_scenario_queue_evolution.png"))
plt.close()

# Figure 2: Utilization Distribution vs Stability Boundary
plt.figure(figsize=(8, 5), dpi=300)
plt.hist(df["utilization"], bins=80, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
plt.axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Stability Limit (ρ = 1.0)")
plt.xlabel("Utilization Ratio (ρ = λ / μ)", fontsize=11)
plt.ylabel("Probability Density", fontsize=11)
plt.title("System Utilization Distribution", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_utilization_distribution.png"))
plt.close()

# Figure 3: Queue Transition Dynamics (Q(t) vs Q(t+1))
plt.figure(figsize=(7, 6), dpi=300)
# Sample 10,000 points to prevent plotting artifact clutter
sample_subset = df.sample(n=10000, random_state=42)
plt.scatter(
    sample_subset["queue_after"], 
    sample_subset["future_queue"], 
    c=sample_subset["utilization"], 
    cmap="coolwarm", 
    alpha=0.4, 
    s=12
)
cbar = plt.colorbar()
cbar.set_label("Utilization (ρ)", fontsize=10)
plt.plot([0, df["queue_after"].max()], [0, df["queue_after"].max()], "k--", alpha=0.7, label="Identity line (Q(t+1) = Q(t))")
plt.xlabel("Current Queue Q(t)", fontsize=11)
plt.ylabel("Next Queue Q(t+1)", fontsize=11)
plt.title("Temporal Queue Transition Dynamics", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_transition_dynamics.png"))
plt.close()

print(f"Verification complete. Generated files and plots saved to '{FIGURES_DIR}' and 'dataset_v2_1/'.")