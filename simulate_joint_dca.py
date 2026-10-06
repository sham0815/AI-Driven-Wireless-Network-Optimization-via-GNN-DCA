import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv

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
print(f"HIGH-STRESS JOINT QUEUE-THEORETIC DCA (Device: {device})")
print("=" * 70)

# Load GNN
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

# Baseline 1: Fixed Static
def allocate_static():
    return np.array([i % NUM_CHANNELS for i in range(NUM_CELLS)])

# Baseline 2: Local Greedy (Myopic reactive)
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

# Novel Joint Optimization: Coordinate Descent over Global Network Cost
def allocate_joint_gnn_dca(pred_q, arrival_rates, adj_matrix, ch_qualities, max_iters=3):
    # Initialize from current greedy state
    alloc = allocate_greedy(pred_q, adj_matrix, ch_qualities)
    
    def compute_network_cost(current_alloc):
        interf = count_interference(current_alloc, adj_matrix)
        mu = calculate_service_capacity(ch_qualities, interf)
        rho = (arrival_rates + pred_q) / mu
        # Global objective: minimize total queue drift and buffer violation penalties
        instability_penalty = np.sum(np.maximum(0.0, rho - 1.0)**2) * 50.0
        drift_penalty = np.sum(pred_q)
        collision_penalty = np.sum(interf) * 2.0
        return instability_penalty + drift_penalty + collision_penalty

    # Iterative coordinate descent to escape local greedy traps
    for _ in range(max_iters):
        improved = False
        for cell in range(NUM_CELLS):
            best_ch = alloc[cell]
            best_cost = compute_network_cost(alloc)
            
            for ch in range(NUM_CHANNELS):
                if ch == alloc[cell]:
                    continue
                temp_alloc = alloc.copy()
                temp_alloc[cell] = ch
                cost = compute_network_cost(temp_alloc)
                if cost < best_cost:
                    best_cost = cost
                    best_ch = ch
                    improved = True
            alloc[cell] = best_ch
        if not improved:
            break
            
    return alloc

# Rollout evaluation on high-stress scenarios
# Pick the scenarios with highest aggregate arrival rates
scenario_loads = df.groupby("scenario")["arrival_rate"].mean()
HEAVY_SCENARIOS = scenario_loads.nlargest(15).index.values
HORIZON_STEPS = 60

metrics = {
    "Static": {"drops": 0.0, "queue_history": [], "congested_steps": 0},
    "Greedy": {"drops": 0.0, "queue_history": [], "congested_steps": 0},
    "Joint_GNN_DCA": {"drops": 0.0, "queue_history": [], "congested_steps": 0}
}

print(f"Simulating {len(HEAVY_SCENARIOS)} High-Stress Scenarios over {HORIZON_STEPS} Steps...")

for scn in HEAVY_SCENARIOS:
    scn_df = df[df["scenario"] == scn].sort_values(["time", "cell_id"])
    adj = adj_matrices[scn]
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long).to(device)
    
    for strat in ["Static", "Greedy", "Joint_GNN_DCA"]:
        t0 = scn_df[scn_df["time"] == 0]
        current_queues = t0["queue_after"].values.copy()
        current_channel = allocate_static()
        
        for t in range(HORIZON_STEPS):
            t_data = scn_df[scn_df["time"] == t]
            arrivals = t_data["arrivals"].values
            arrival_rate = t_data["arrival_rate"].values
            ch_qual = t_data["channel_quality"].values
            
            interf_prev = count_interference(current_channel, adj)
            mu_prev = calculate_service_capacity(ch_qual, interf_prev)
            
            if strat == "Static":
                alloc = allocate_static()
            elif strat == "Greedy":
                alloc = allocate_greedy(current_queues, adj, ch_qual)
            elif strat == "Joint_GNN_DCA":
                # Online feature synthesis
                rho_est = arrival_rate / mu_prev
                delay_est = current_queues / (mu_prev + 1e-4)
                
                # Synthetic online feature vector
                online_feats = np.column_stack([
                    current_queues,                # queue_before
                    current_queues,                # queue_after
                    arrival_rate,                  # base_arrival_rate
                    arrival_rate,                  # arrival_rate
                    arrivals,                      # arrivals
                    current_channel,               # channel
                    ch_qual,                       # channel_quality
                    interf_prev,                   # interference
                    mu_prev,                       # service_capacity
                    np.minimum(current_queues, mu_prev), # packets_served
                    rho_est,                       # utilization
                    delay_est,                     # estimated_delay
                    np.zeros(NUM_CELLS)            # packet_drops
                ])
                
                norm_feats = (online_feats - feature_mean) / feature_scale
                x_t = torch.tensor(norm_feats, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    pred_q = gnn_model(x_t, edge_index).clamp(min=0.0).cpu().numpy()
                    
                alloc = allocate_joint_gnn_dca(pred_q, arrival_rate, adj, ch_qual)
            
            current_channel = alloc
            interf = count_interference(alloc, adj)
            mu = calculate_service_capacity(ch_qual, interf)
            
            # State transition
            new_queues = np.maximum(0.0, current_queues + arrivals - mu)
            drops = np.maximum(0.0, new_queues - BUFFER_CAPACITY)
            current_queues = np.minimum(BUFFER_CAPACITY, new_queues)
            
            metrics[strat]["drops"] += np.sum(drops)
            metrics[strat]["queue_history"].append(np.mean(current_queues))
            metrics[strat]["congested_steps"] += np.sum((arrival_rate / mu) > 1.0)

print("\n" + "=" * 70)
print(f"HIGH-STRESS BENCHMARK RESULTS ({len(HEAVY_SCENARIOS)} Scenarios x {HORIZON_STEPS} Steps)")
print("=" * 70)
print(f"{'Strategy':<20} | {'Total Drops':<12} | {'Mean Queue':<14} | {'Congested Steps (ρ>1)':<22}")
print("-" * 75)

for strat in metrics:
    d = metrics[strat]["drops"]
    q = np.mean(metrics[strat]["queue_history"])
    c = metrics[strat]["congested_steps"]
    print(f"{strat:<20} | {d:<12.1f} | {q:<14.3f} | {c:<22}")

# Plotting
plt.figure(figsize=(9, 5), dpi=300)
for strat in metrics:
    arr = np.array(metrics[strat]["queue_history"]).reshape(len(HEAVY_SCENARIOS), HORIZON_STEPS)
    plt.plot(np.mean(arr, axis=0), label=strat, linewidth=2.0)

plt.xlabel("Continuous Horizon Timestep", fontsize=11)
plt.ylabel("Mean Network Queue Length", fontsize=11)
plt.title("High-Stress Closed-Loop Queue Trajectory", fontsize=12)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig5_high_stress_dca_performance.png")
plt.savefig(fig_path)
plt.close()

print(f"\nFigure saved to: {fig_path}")
print("=" * 70)