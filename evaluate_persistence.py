import os
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

OUTPUT_DIR = "graph_dataset"
TEST_GRAPH_PATH = os.path.join(OUTPUT_DIR, "test_graphs.pt")
MEAN_PATH = os.path.join(OUTPUT_DIR, "feature_mean.npy")
SCALE_PATH = os.path.join(OUTPUT_DIR, "feature_scale.npy")

print("=" * 70)
print("EVALUATING BASELINE 1: PERSISTENCE MODEL ON TEST SCENARIOS")
print("=" * 70)

# Load scaler parameters to recover original queue values
feature_mean = np.load(MEAN_PATH)
feature_scale = np.load(SCALE_PATH)

# Index 1 = queue_after in FEATURE_COLUMNS
QUEUE_AFTER_IDX = 1
q_mean = float(feature_mean[QUEUE_AFTER_IDX])
q_scale = float(feature_scale[QUEUE_AFTER_IDX])

print(f"Loading test set from: {TEST_GRAPH_PATH}...")
test_graphs = torch.load(TEST_GRAPH_PATH, weights_only=False)
test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

y_true_list = []
y_pred_list = []

for batch in test_loader:
    # Ground truth future_queue at t+1
    y_actual = batch.y.cpu().numpy()
    
    # Invert z-score to recover actual raw queue_after at time t
    norm_q = batch.x[:, QUEUE_AFTER_IDX].cpu().numpy()
    raw_q = (norm_q * q_scale) + q_mean
    
    # Persistence: predict Q(t+1) = Q(t)
    y_pred = raw_q
    
    y_true_list.extend(y_actual)
    y_pred_list.extend(y_pred)

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)

# Global metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- GLOBAL TEST PERFORMANCE (PERSISTENCE) ---")
print(f"Total Test Samples: {len(y_true):,}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Congested regime metrics (queue > 20 or top 10% threshold)
congestion_thresh = 20.0
cong_mask = y_true >= congestion_thresh

mae_cong = mean_absolute_error(y_true[cong_mask], y_pred[cong_mask])
rmse_cong = np.sqrt(mean_squared_error(y_true[cong_mask], y_pred[cong_mask]))

print(f"\n--- CONGESTED REGIME EVALUATION (Actual Q(t+1) >= {congestion_thresh}) ---")
print(f"Congested Samples : {cong_mask.sum():,} ({100 * cong_mask.sum() / len(y_true):.2f}%)")
print(f"Congested MAE     : {mae_cong:.4f}")
print(f"Congested RMSE    : {rmse_cong:.4f}")
print("=" * 70)