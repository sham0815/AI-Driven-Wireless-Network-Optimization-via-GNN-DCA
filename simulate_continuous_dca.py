import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv

# ============================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================
OUTPUT_DIR = "graph_dataset"
MODEL_PATH = "saved_models/best_gnn_model.pt"
FIGURES_DIR = os.path.join("dataset_v2_1", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join("dataset_v2_1", "wireless_queue_dataset_v2_1.csv")
ADJ_PATH = os.path.join("dataset_v2_1", "adjacency_matrices.npy")
MEAN_PATH = os.path.join(OUTPUT_DIR, "feature_mean.npy")
SCALE_PATH = os.path.join(OUTPUT_DIR, "feature_scale.npy")

NUM_CELLS = 20
NUM_CHANNELS = 5
BUFFER_CAPACITY = 200.0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 70)
print(f"CLOSED-LOOP MULTI-STEP DCA SIMULATION (Device: {device})")
print("=" * 70)

# ============================================================
# 1. LOAD MODEL & NORMALIZERS
# ============================================================
class WirelessGraphSAGE(nn.Module):
    def __init__(self, in_features=13, hidden_dim=64):
        super(WirelessGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_features, hidden_dim, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index):
        h = self.relu1(self.bn1(self.conv1(x, edge_index)))
        h = self.relu2(self.bn2(self.conv2(h, edge_index)))
        return self.head(h).squeeze(-1)

gnn_model = WirelessGraphSAGE(in_features=13, hidden_dim=64).to(device)
gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
gnn_model.eval()

feature_mean = np.load(MEAN_PATH)
feature_scale = np.load(SCALE_PATH)

df = pd.read_csv(CSV_PATH)
adj_matrices = np.load(ADJ_PATH)

# ============================================================
# 2. PHYSICAL CHANNEL & CAPACITY HELPERS
# ============================================================
def calculate_service_capacity(channel_quality, interference):
    base_cap = 20.0 * channel_quality
    penalty = 1.0 + (0.35 * interference)
    return np.maximum(1.0, base_cap / penalty)

def count_interference(allocations, adj_matrix):
    interf = np.zeros(NUM_CELLS)
    for i in range(NUM_CELLS):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        interf[i] = np.sum(allocations[neighbors] == allocations[i])
    return interf

# ============================================================
# 3. ALLOCATION POLICIES
# ============================================================
def allocate_static():
    return np.array([i % NUM_CHANNELS for i in range(NUM_CELLS)])

def allocate_greedy(current_queues, adj_matrix, ch_qualities):
    sorted_cells = np.argsort(-current_queues)
    alloc = np.zeros(NUM_CELLS, dtype=int)
    for cell in sorted_cells:
        best_ch = 0
        min_cost = float("inf")
        for ch in range(NUM_CHANNELS):
            temp = alloc.copy()
            temp[cell] = ch
            interf = count_interference(temp, adj_matrix)[cell]
            cost = interf - (0.15 * ch_qualities[cell])
            if cost < min_cost:
                min_cost = cost
                best_ch = ch
        alloc[cell] = best_ch
    return alloc

def allocate_gnn_queue_theoretic(pred_q, current_q, arrival_rates, adj_matrix, ch_qualities):
    # Anticipate future surge: lambda_est + predicted delta queue
    anticipated_load = arrival_rates + np.maximum(0.0, pred_q - current_q)
    sorted_cells = np.argsort(-anticipated_load)
    alloc = np.zeros(NUM_CELLS, dtype=int)
    
    for cell in sorted_cells:
        best_ch = 0
        lowest_penalty = float("inf")
        for ch in range(NUM_CHANNELS):
            temp = alloc.copy()
            temp[cell] = ch
            interf = count_interference(temp, adj_matrix)[cell]
            mu = calculate_service_capacity(ch_qualities[cell], interf)
            
            rho_hat = anticipated_load[cell] / mu
            # Heavy quadratic penalty for queue instability boundary (rho > 1.0)
            penalty = 150.0 * max(0.0, rho_hat - 1.0)**2 + rho_hat
            
            if penalty < lowest_penalty:
                lowest_penalty = penalty
                best_ch = ch
        alloc[cell] = best_ch
    return alloc

# ============================================================
# 4. MULTI-STEP CLOSED-LOOP ROLLOUT (50 TIMESTEPS PER SCENARIO)
# ============================================================
# Evaluate across 10 unseen test scenarios
TEST_SCENARIOS = df["scenario"].unique()[-10:]
HORIZON_STEPS = 50

metrics = {
    "Static": {"drops": 0.0, "queue_history": [], "congested_steps": 0},
    "Greedy": {"drops": 0.0, "queue_history": [], "congested_steps": 0},
    "GNN_Queue_Guided": {"drops": 0.0, "queue_history": [], "congested_steps": 0}
}

print(f"Simulating {len(TEST_SCENARIOS)} scenarios over {HORIZON_STEPS} continuous timesteps...")

for scn in TEST_SCENARIOS:
    scn_df = df[df["scenario"] == scn].sort_values(["time", "cell_id"])
    adj = adj_matrices[scn]
    
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long).to(device)
    
    # Run separate rollout per strategy starting from identical initial conditions
    for strat in ["Static", "Greedy", "GNN_Queue_Guided"]:
        t0_data = scn_df[scn_df["time"] == 0]
        current_queues = t0_data["queue_after"].values.copy()
        
        for t in range(HORIZON_STEPS):
            t_data = scn_df[scn_df["time"] == t]
            arrivals = t_data["arrivals"].values
            arrival_rate = t_data["arrival_rate"].values
            ch_qual = t_data["channel_quality"].values
            
            if strat == "Static":
                alloc = allocate_static()
            elif strat == "Greedy":
                alloc = allocate_greedy(current_queues, adj, ch_qual)
            elif strat == "GNN_Queue_Guided":
                # Construct feature representation using current state
                feats = t_data[[
                    "queue_before", "queue_after", "base_arrival_rate", "arrival_rate",
                    "arrivals", "channel", "channel_quality", "interference",
                    "service_capacity", "packets_served", "utilization",
                    "estimated_delay", "packet_drops"
                ]].values.copy()
                feats[:, 1] = current_queues  # Inject live state
                
                # Normalize features
                norm_feats = (feats - feature_mean) / feature_scale
                x_t = torch.tensor(norm_feats, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    pred_q = gnn_model(x_t, edge_index).clamp(min=0.0).cpu().numpy()
                    
                alloc = allocate_gnn_queue_theoretic(pred_q, current_queues, arrival_rate, adj, ch_qual)
            
            # Physics execution
            interf = count_interference(alloc, adj)
            mu = calculate_service_capacity(ch_qual, interf)
            
            unserved = current_queues + arrivals - mu
            new_queues = np.maximum(0.0, unserved)
            drops = np.maximum(0.0, new_queues - BUFFER_CAPACITY)
            current_queues = np.minimum(BUFFER_CAPACITY, new_queues)
            
            metrics[strat]["drops"] += np.sum(drops)
            metrics[strat]["queue_history"].append(np.mean(current_queues))
            metrics[strat]["congested_steps"] += np.sum((arrival_rate / mu) > 1.0)

# ============================================================
# 5. RESULTS & VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print(f"MULTI-STEP CLOSED-LOOP PERFORMANCE ({len(TEST_SCENARIOS)} Scenarios x {HORIZON_STEPS} Steps)")
print("=" * 70)
print(f"{'Strategy':<20} | {'Total Drops':<12} | {'Mean Queue':<14} | {'Congested Instances (ρ>1)':<22}")
print("-" * 75)

for strat in metrics:
    d = metrics[strat]["drops"]
    q = np.mean(metrics[strat]["queue_history"])
    c = metrics[strat]["congested_steps"]
    print(f"{strat:<20} | {d:<12.1f} | {q:<14.3f} | {c:<22}")

# Plotting average queue evolution
plt.figure(figsize=(9, 5), dpi=300)
for strat in metrics:
    # Reshape and average across scenarios
    arr = np.array(metrics[strat]["queue_history"]).reshape(len(TEST_SCENARIOS), HORIZON_STEPS)
    mean_trajectory = np.mean(arr, axis=0)
    plt.plot(mean_trajectory, label=strat, linewidth=1.8)

plt.xlabel("Continuous Horizon Timestep", fontsize=11)
plt.ylabel("Mean Network Queue Length", fontsize=11)
plt.title("Closed-Loop Dynamic Allocation: Queue Evolution Over Time", fontsize=12)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
fig_out = os.path.join(FIGURES_DIR, "fig4_continuous_allocation_performance.png")
plt.savefig(fig_out)
plt.close()

print(f"\nSaved queue evolution plot to: {fig_out}")
print("=" * 70)