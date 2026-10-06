import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ============================================================
# GNN GRAPH DATASET CONSTRUCTION
# Queue-Theory Guided Dynamic Channel Allocation
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

DATASET_DIR = "dataset_v2_1"

CSV_PATH = os.path.join(
    DATASET_DIR,
    "wireless_queue_dataset_v2_1.csv"
)

ADJ_PATH = os.path.join(
    DATASET_DIR,
    "adjacency_matrices.npy"
)

OUTPUT_DIR = "graph_dataset"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING V2.1 DATASET")
print("=" * 70)

df = pd.read_csv(
    CSV_PATH
)

adjacency_matrices = np.load(
    ADJ_PATH
)

print(
    f"\nCSV shape: {df.shape}"
)

print(
    f"Adjacency shape: "
    f"{adjacency_matrices.shape}"
)


# ============================================================
# 3. BASIC PARAMETERS
# ============================================================

NUM_SCENARIOS = (
    df["scenario"].nunique()
)

NUM_CELLS = (
    df["cell_id"].nunique()
)

NUM_CHANNELS = 5


print(
    f"\nNumber of scenarios: "
    f"{NUM_SCENARIOS}"
)

print(
    f"Number of cells: "
    f"{NUM_CELLS}"
)


# ============================================================
# 4. SCENARIO-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING SCENARIO-LEVEL SPLIT")
print("=" * 70)


# IMPORTANT:
# We split entire scenarios rather than individual rows.
#
# This prevents information from the same network appearing
# in both training and testing.


all_scenarios = np.arange(
    NUM_SCENARIOS
)

# Reproducible shuffle
rng = np.random.default_rng(
    42
)

rng.shuffle(
    all_scenarios
)


# ------------------------------------------------------------
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ------------------------------------------------------------

train_end = int(
    0.70 * NUM_SCENARIOS
)

val_end = int(
    0.85 * NUM_SCENARIOS
)


train_scenarios = (
    all_scenarios[:train_end]
)

val_scenarios = (
    all_scenarios[
        train_end:val_end
    ]
)

test_scenarios = (
    all_scenarios[
        val_end:
    ]
)


print(
    f"\nTraining scenarios: "
    f"{len(train_scenarios)}"
)

print(
    f"Validation scenarios: "
    f"{len(val_scenarios)}"
)

print(
    f"Testing scenarios: "
    f"{len(test_scenarios)}"
)


# ============================================================
# 5. NODE FEATURES
# ============================================================

# These are the features available to the GNN at time t.
#
# IMPORTANT:
#
# future_queue is NOT included as an input.
#
# It is the prediction target.


FEATURE_COLUMNS = [

    "queue_before",

    "queue_after",

    "base_arrival_rate",

    "arrival_rate",

    "arrivals",

    "channel",

    "channel_quality",

    "interference",

    "service_capacity",

    "packets_served",

    "utilization",

    "estimated_delay",

    "packet_drops"
]


print("\n" + "=" * 70)
print("NODE FEATURES")
print("=" * 70)

for i, feature in enumerate(
    FEATURE_COLUMNS
):

    print(
        f"{i}: {feature}"
    )


print(
    f"\nNumber of node features: "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# 6. FEATURE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING FEATURES")
print("=" * 70)


# Fit scaler ONLY on training scenarios.
#
# This is important because using validation/test data to
# calculate normalization statistics would cause data leakage.


train_df = df[
    df["scenario"].isin(
        train_scenarios
    )
]


scaler = StandardScaler()

scaler.fit(
    train_df[
        FEATURE_COLUMNS
    ].values
)


# Transform entire dataset using the training scaler

df[
    FEATURE_COLUMNS
] = scaler.transform(
    df[
        FEATURE_COLUMNS
    ].values
)


# Save scaler parameters

np.save(
    os.path.join(
        OUTPUT_DIR,
        "feature_mean.npy"
    ),
    scaler.mean_
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "feature_scale.npy"
    ),
    scaler.scale_
)


print(
    "\nFeature normalization complete."
)


# ============================================================
# 7. CREATE EDGE INDEX
# ============================================================

def adjacency_to_edge_index(
    adjacency
):

    """
    Convert adjacency matrix to
    PyTorch Geometric edge_index.
    """

    rows, cols = np.nonzero(
        adjacency
    )

    edge_index = torch.tensor(
        np.vstack(
            [rows, cols]
        ),
        dtype=torch.long
    )

    return edge_index


# ============================================================
# 8. CREATE GRAPH FROM ONE TIMESTEP
# ============================================================

def create_graph(
    scenario,
    time,
    scenario_df,
    adjacency
):

    """
    Create one temporal graph.

    Nodes:
        20 cells

    Edges:
        Neighboring cells

    Node features:
        Current state at time t

    Target:
        Future queue at time t+1
    """

    # --------------------------------------------------------
    # Make sure cells are in correct order
    # --------------------------------------------------------

    timestep_df = (
        scenario_df[
            scenario_df["time"] == time
        ]
        .sort_values(
            "cell_id"
        )
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(timestep_df) != NUM_CELLS:

        raise ValueError(
            f"Scenario {scenario}, "
            f"time {time} has "
            f"{len(timestep_df)} cells "
            f"instead of {NUM_CELLS}."
        )


    # --------------------------------------------------------
    # Node feature matrix
    #
    # Shape:
    #
    # [20, 13]
    # --------------------------------------------------------

    x = torch.tensor(
        timestep_df[
            FEATURE_COLUMNS
        ].values,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Target
    #
    # Shape:
    #
    # [20]
    # --------------------------------------------------------

    y = torch.tensor(
        timestep_df[
            "future_queue"
        ].values,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Graph edges
    # --------------------------------------------------------

    edge_index = (
        adjacency_to_edge_index(
            adjacency
        )
    )


    # --------------------------------------------------------
    # Additional useful information
    # --------------------------------------------------------

    scenario_tensor = torch.tensor(
        [scenario],
        dtype=torch.long
    )

    time_tensor = torch.tensor(
        [time],
        dtype=torch.long
    )


    # --------------------------------------------------------
    # Create PyTorch Geometric Data object
    # --------------------------------------------------------

    graph = Data(

        x=x,

        edge_index=edge_index,

        y=y,

        scenario=scenario_tensor,

        time=time_tensor
    )


    return graph


# ============================================================
# 9. BUILD DATASET
# ============================================================

def build_graph_dataset(
    scenario_list,
    dataset_df,
    adjacency_matrices,
    dataset_name
):

    print(
        f"\nBuilding {dataset_name} graphs..."
    )

    graphs = []

    for count, scenario in enumerate(
        scenario_list
    ):

        scenario_df = dataset_df[
            dataset_df["scenario"] == scenario
        ]

        adjacency = (
            adjacency_matrices[
                scenario
            ]
        )

        # ----------------------------------------------------
        # Get available timesteps
        # ----------------------------------------------------

        timesteps = sorted(
            scenario_df[
                "time"
            ].unique()
        )


        for time in timesteps:

            graph = create_graph(
                scenario,
                time,
                scenario_df,
                adjacency
            )

            graphs.append(
                graph
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (count + 1) % 25 == 0
            or count == 0
            or count + 1 == len(
                scenario_list
            )
        ):

            print(
                f"  Processed "
                f"{count + 1}/"
                f"{len(scenario_list)} "
                f"scenarios"
            )


    print(
        f"\n{dataset_name} graphs created: "
        f"{len(graphs):,}"
    )

    return graphs


# ============================================================
# 10. BUILD TRAINING GRAPHS
# ============================================================

train_graphs = build_graph_dataset(
    train_scenarios,
    df,
    adjacency_matrices,
    "TRAIN"
)


# ============================================================
# 11. BUILD VALIDATION GRAPHS
# ============================================================

val_graphs = build_graph_dataset(
    val_scenarios,
    df,
    adjacency_matrices,
    "VALIDATION"
)


# ============================================================
# 12. BUILD TEST GRAPHS
# ============================================================

test_graphs = build_graph_dataset(
    test_scenarios,
    df,
    adjacency_matrices,
    "TEST"
)


# ============================================================
# 13. SAVE GRAPH DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING GRAPH DATASETS")
print("=" * 70)


torch.save(
    train_graphs,
    os.path.join(
        OUTPUT_DIR,
        "train_graphs.pt"
    )
)


torch.save(
    val_graphs,
    os.path.join(
        OUTPUT_DIR,
        "val_graphs.pt"
    )
)


torch.save(
    test_graphs,
    os.path.join(
        OUTPUT_DIR,
        "test_graphs.pt"
    )
)


# ============================================================
# 14. CREATE DATA LOADERS
# ============================================================

print(
    "\nCreating PyTorch Geometric DataLoaders..."
)


BATCH_SIZE = 32


train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 15. SAVE DATASET INFORMATION
# ============================================================

dataset_info = {

    "num_scenarios":
        NUM_SCENARIOS,

    "num_cells":
        NUM_CELLS,

    "num_channels":
        NUM_CHANNELS,

    "num_node_features":
        len(FEATURE_COLUMNS),

    "train_scenarios":
        len(train_scenarios),

    "validation_scenarios":
        len(val_scenarios),

    "test_scenarios":
        len(test_scenarios),

    "train_graphs":
        len(train_graphs),

    "validation_graphs":
        len(val_graphs),

    "test_graphs":
        len(test_graphs),

    "batch_size":
        BATCH_SIZE
}


# Save readable information

with open(
    os.path.join(
        OUTPUT_DIR,
        "dataset_info.txt"
    ),
    "w"
) as f:

    for key, value in (
        dataset_info.items()
    ):

        f.write(
            f"{key}: {value}\n"
        )


# ============================================================
# 16. VERIFY ONE GRAPH
# ============================================================

print("\n" + "=" * 70)
print("GRAPH VERIFICATION")
print("=" * 70)


sample_graph = train_graphs[0]


print(
    "\nSample graph:"
)

print(
    sample_graph
)


print(
    "\nNode feature shape:"
)

print(
    sample_graph.x.shape
)


print(
    "\nEdge index shape:"
)

print(
    sample_graph.edge_index.shape
)


print(
    "\nTarget shape:"
)

print(
    sample_graph.y.shape
)


print(
    "\nNumber of nodes:"
)

print(
    sample_graph.num_nodes
)


print(
    "\nNumber of edges:"
)

print(
    sample_graph.num_edges
)


# ============================================================
# 17. CHECK FOR NaN / INFINITY
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)


has_nan = False

has_inf = False


for graph in [
    sample_graph
]:

    if torch.isnan(
        graph.x
    ).any():

        has_nan = True

    if torch.isinf(
        graph.x
    ).any():

        has_inf = True


    if torch.isnan(
        graph.y
    ).any():

        has_nan = True

    if torch.isinf(
        graph.y
    ).any():

        has_inf = True


print(
    f"\nNaN detected: "
    f"{has_nan}"
)

print(
    f"Infinity detected: "
    f"{has_inf}"
)


# ============================================================
# 18. TEST DATALOADER
# ============================================================

print("\n" + "=" * 70)
print("DATALOADER TEST")
print("=" * 70)


batch = next(
    iter(train_loader)
)


print(
    "\nBatch:"
)

print(
    batch
)


print(
    "\nBatch node features:"
)

print(
    batch.x.shape
)


print(
    "\nBatch targets:"
)

print(
    batch.y.shape
)


print(
    "\nBatch edge index:"
)

print(
    batch.edge_index.shape
)


print(
    "\nBatch assignment vector:"
)

print(
    batch.batch.shape
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH DATASET CREATION COMPLETE")
print("=" * 70)


print(
    f"""
Dataset summary:

Original CSV:
    {len(df):,} rows

Scenarios:
    {NUM_SCENARIOS}

Cells per graph:
    {NUM_CELLS}

Channels:
    {NUM_CHANNELS}

Node features:
    {len(FEATURE_COLUMNS)}

Training graphs:
    {len(train_graphs):,}

Validation graphs:
    {len(val_graphs):,}

Testing graphs:
    {len(test_graphs):,}

Batch size:
    {BATCH_SIZE}

Output directory:
    {OUTPUT_DIR}/
"""
)


print(
    "\nFiles created:"
)

print(
    "  train_graphs.pt"
)

print(
    "  val_graphs.pt"
)

print(
    "  test_graphs.pt"
)

print(
    "  feature_mean.npy"
)

print(
    "  feature_scale.npy"
)

print(
    "  dataset_info.txt"
)


print(
    "\nReady for GNN model training."
)