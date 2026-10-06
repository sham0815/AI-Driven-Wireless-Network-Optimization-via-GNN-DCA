import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset artifacts
DATASET_DIR = "dataset_v2_1"
ADJ_PATH = os.path.join(DATASET_DIR, "adjacency_matrices.npy")
POS_PATH = os.path.join(DATASET_DIR, "positions.npy") # If positions exist, or we can extract from CSV

adjacency_matrices = np.load(ADJ_PATH)

# Pick Scenario 0 to visualize
sample_adj = adjacency_matrices[0]

# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(sample_adj)

print(f"Visualizing Scenario 0 Graph:")
print(f"Nodes (Cells): {G.number_of_nodes()}")
print(f"Edges (Interference Links): {G.number_of_edges()}")

# Plotting
plt.figure(figsize=(8, 8), dpi=300)
pos = nx.spring_layout(G, seed=42) # Computes clean spatial node layout

nx.draw_networkx_nodes(G, pos, node_size=700, node_color="#1f77b4", alpha=0.9)
nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray", alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", font_weight="bold")

plt.title("Wireless Network Graph Structure (Scenario 0: 20 Cells, Inter-cell Edges)", fontsize=11)
plt.axis("off")
plt.tight_layout()

output_fig = os.path.join(DATASET_DIR, "figures", "network_graph_visualization.png")
plt.savefig(output_fig)
plt.show()

print(f"\nGraph visualization saved to: {output_fig}")