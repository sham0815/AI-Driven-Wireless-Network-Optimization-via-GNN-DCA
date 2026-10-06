import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

# ============================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================
OUTPUT_DIR = "graph_dataset"
MODEL_PATH = "saved_models/best_gnn_model.pt"
TEST_GRAPH_PATH = os.path.join(OUTPUT_DIR, "test_graphs.pt")
MEAN_PATH = os.path.join(OUTPUT_DIR, "feature_mean.npy")
SCALE_PATH = os.path.join(OUTPUT_DIR, "feature_scale.npy")

NUM_CELLS = 20
NUM_CHANNELS = 5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 70)
print("DYNAMIC CHANNEL ALLOCATION (DCA) SIMULATION & BENCHMARKING")
print("=" * 70)

# ============================================================
# 1. LOAD MODEL & NORMALIZATION ARTIFACTS
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

test_graphs = torch.load(TEST_GRAPH_PATH, weights_only=False)

# ============================================================
# 2. WIRELESS SIMULATION SUBROUTINES
# ============================================================
def calculate_service_capacity(channel_quality, interference):
    # Matches physical simulation degradation: mu degrades with co-channel collisions
    base_cap = 20.0 * channel_quality
    penalty = 1.0 + (0.35 * interference)
    return max(1.0, base_cap / penalty)

def count_interference(allocations, adj_matrix):
    interf = np.zeros(NUM_CELLS)
    for i in range(NUM_CELLS):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        # Co-channel conflict: neighbor uses exact same channel
        interf[i] = np.sum(allocations[neighbors] == allocations[i])
    return interf

# ============================================================
# 3. ALLOCATION STRATEGIES
# ============================================================
def run_static_allocation():
    # Fixed assignment: cell_id % NUM_CHANNELS
    return np.array([i % NUM_CHANNELS for i in range(NUM_CELLS)])

def run_greedy_current_state(current_queues, adj_matrix, channel_qualities):
    # Allocate to highest current queue first, minimizing immediate collisions
    sorted_cells = np.argsort(-current_queues)
    allocations = np.zeros(NUM_CELLS, dtype=int)
    assigned = set()
    
    for cell in sorted_cells:
        best_ch = 0
        min_cost = float("inf")
        for ch in range(NUM_CHANNELS):
            temp_alloc = allocations.copy()
            temp_alloc[cell] = ch
            interf = count_interference(temp_alloc, adj_matrix)[cell]
            # Cost balances capacity against interference
            cost = interf - (0.1 * channel_qualities[cell])
            if cost < min_cost:
                min_cost = cost
                best_ch = ch
        allocations[cell] = best_ch
        assigned.add(cell)
    return allocations

def run_queue_theory_gnn_allocation(pred_future_queues, arrival_rates, adj_matrix, channel_qualities):
    # Sort by projected queue pressure: rho_hat = lambda_hat / mu_base
    projected_demand = pred_future_queues + arrival_rates
    sorted_cells = np.argsort(-projected_demand)
    allocations = np.zeros(NUM_CELLS, dtype=int)
    
    for cell in sorted_cells:
        best_ch = 0
        lowest_utilization_penalty = float("inf")
        
        for ch in range(NUM_CHANNELS):
            temp_alloc = allocations.copy()
            temp_alloc[cell] = ch
            interf = count_interference(temp_alloc, adj_matrix)[cell]
            mu = calculate_service_capacity(channel_qualities[cell], interf)
            
            # Queue-theoretic objective: enforce rho <= 1.0 boundary
            rho_hat = projected_demand[cell] / mu
            penalty = max(0.0, rho_hat - 1.0) * 100.0 + rho_hat
            
            if penalty < lowest_utilization_penalty:
                lowest_utilization_penalty = penalty
                best_ch = ch
                
        allocations[cell] = best_ch
    return allocations

# ============================================================
# 4. BENCHMARK SIMULATION LOOP (EVALUATING OVER TEST GRAPHS)
# ============================================================
print("\nRunning multi-strategy allocation comparison over 200 test horizons...")

EVAL_SAMPLES = min(200, len(test_graphs))
results = {
    "Static": {"drops": 0, "avg_queue": 0.0, "congested_count": 0},
    "Greedy_Current": {"drops": 0, "avg_queue": 0.0, "congested_count": 0},
    "GNN_Queue_Guided": {"drops": 0, "avg_queue": 0.0, "congested_count": 0}
}

for idx in range(EVAL_SAMPLES):
    graph = test_graphs[idx]
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    
    # Reconstruct unnormalized features
    x_raw = (graph.x.numpy() * feature_scale) + feature_mean
    current_q = x_raw[:, 1]
    arrival_r = x_raw[:, 3]
    ch_qual = x_raw[:, 6]
    
    # Adjacency matrix from edge_index
    adj = np.zeros((NUM_CELLS, NUM_CELLS))
    edges = graph.edge_index.numpy()
    adj[edges[0], edges[1]] = 1
    
    # GNN Inference
    with torch.no_grad():
        gnn_pred = gnn_model(x, edge_index).clamp(min=0.0).cpu().numpy()
        
    strategies = {
        "Static": run_static_allocation(),
        "Greedy_Current": run_greedy_current_state(current_q, adj, ch_qual),
        "GNN_Queue_Guided": run_queue_theory_gnn_allocation(gnn_pred, arrival_r, adj, ch_qual)
    }
    
    for name, ch_alloc in strategies.items():
        interf = count_interference(ch_alloc, adj)
        service_caps = np.array([calculate_service_capacity(ch_qual[i], interf[i]) for i in range(NUM_CELLS)])
        
        # Simulate queue transition
        next_q = np.maximum(0.0, current_q + arrival_r - service_caps)
        # Drop threshold = 200 buffer size
        drops = np.maximum(0.0, next_q - 200.0)
        final_q = np.minimum(200.0, next_q)
        
        results[name]["drops"] += np.sum(drops)
        results[name]["avg_queue"] += np.mean(final_q)
        results[name]["congested_count"] += np.sum((arrival_r / service_caps) > 1.0)

# ============================================================
# 5. PRESENTATION OF RESULTS
# ============================================================
print("\n" + "=" * 70)
print(f"ALLOCATION PERFORMANCE METRICS (Averaged across {EVAL_SAMPLES} Horizons)")
print("=" * 70)

print(f"{'Strategy':<20} | {'Total Drops':<12} | {'Mean Cell Queue':<16} | {'Congested Cells (ρ>1)':<20}")
print("-" * 75)

for name in results:
    total_drops = results[name]["drops"]
    mean_queue = results[name]["avg_queue"] / EVAL_SAMPLES
    congested_cells = results[name]["congested_count"]
    print(f"{name:<20} | {total_drops:<12.1f} | {mean_queue:<16.3f} | {congested_cells:<20}")

print("=" * 70)