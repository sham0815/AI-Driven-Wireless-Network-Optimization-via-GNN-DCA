import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURATION & DEVICE
# ============================================================
OUTPUT_DIR = "graph_dataset"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 70)
print(f"TRAINING MODEL 3: SPATIAL GRAPHSAGE GNN (Device: {device})")
print("=" * 70)

# ============================================================
# 1. LOAD DATASETS
# ============================================================
print("Loading datasets...")
train_graphs = torch.load(os.path.join(OUTPUT_DIR, "train_graphs.pt"), weights_only=False)
val_graphs = torch.load(os.path.join(OUTPUT_DIR, "val_graphs.pt"), weights_only=False)
test_graphs = torch.load(os.path.join(OUTPUT_DIR, "test_graphs.pt"), weights_only=False)

BATCH_SIZE = 64
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 2. DEFINE GRAPHSAGE ARCHITECTURE
# ============================================================
class WirelessGraphSAGE(nn.Module):
    def __init__(self, in_features=13, hidden_dim=64):
        super(WirelessGraphSAGE, self).__init__()
        # Layer 1: Aggregates immediate 1-hop wireless neighbors
        self.conv1 = SAGEConv(in_features, hidden_dim, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Aggregates 2-hop neighbor interference patterns
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        
        # Prediction Head: Maps graph-aggregated embeddings to scalar future_queue
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.relu1(h)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = self.relu2(h)
        
        out = self.head(h)
        return out.squeeze(-1)

model = WirelessGraphSAGE(in_features=13, hidden_dim=64).to(device)

criterion = nn.HuberLoss(delta=1.0)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# ============================================================
# 3. TRAINING LOOP
# ============================================================
EPOCHS = 15
best_val_loss = float("inf")
best_model_path = os.path.join(MODEL_DIR, "best_gnn_model.pt")

print(f"\nStarting training for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        y = batch.y.to(device)
        
        optimizer.zero_grad()
        preds = model(x, edge_index)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch.num_nodes
        
    train_loss /= (len(train_graphs) * 20)
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            y = batch.y.to(device)
            preds = model(x, edge_index)
            loss = criterion(preds, y)
            val_loss += loss.item() * batch.num_nodes
            
    val_loss /= (len(val_graphs) * 20)
    scheduler.step(val_loss)
    elapsed = time.time() - start_time
    
    # Save checkpoint
    saved_str = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        saved_str = "[MODEL SAVED]"
        
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s {saved_str}")

# ============================================================
# 4. TEST SET EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("EVALUATING GNN ON UNSEEN TEST SCENARIOS")
print("=" * 70)

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

y_true_all = []
y_pred_all = []

with torch.no_grad():
    for batch in test_loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        preds = model(x, edge_index)
        preds = torch.clamp(preds, min=0.0)
        
        y_true_all.extend(batch.y.cpu().numpy())
        y_pred_all.extend(preds.cpu().numpy())

y_true = np.array(y_true_all)
y_pred = np.array(y_pred_all)

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- GLOBAL TEST PERFORMANCE (GNN) ---")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Congested regime evaluation
congestion_thresh = 20.0
cong_mask = y_true >= congestion_thresh

mae_cong = mean_absolute_error(y_true[cong_mask], y_pred[cong_mask])
rmse_cong = np.sqrt(mean_squared_error(y_true[cong_mask], y_pred[cong_mask]))

print(f"\n--- CONGESTED REGIME EVALUATION (Actual Q(t+1) >= {congestion_thresh}) ---")
print(f"Congested MAE     : {mae_cong:.4f}")
print(f"Congested RMSE    : {rmse_cong:.4f}")
print("=" * 70)