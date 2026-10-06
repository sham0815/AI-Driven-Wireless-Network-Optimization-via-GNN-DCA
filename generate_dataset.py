import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# QUEUE-THEORY GUIDED DYNAMIC CHANNEL ALLOCATION
#
# DATASET GENERATOR - VERSION 2.1
# ============================================================

np.random.seed(42)


# ============================================================
# 1. SIMULATION PARAMETERS
# ============================================================

NUM_SCENARIOS = 500

NUM_CELLS = 20

NUM_CHANNELS = 5

TIME_STEPS = 200

AREA_SIZE = 1000

NEIGHBOR_DISTANCE = 350

QUEUE_CAPACITY = 200

# V2.1: Slightly lower service capacity
BASE_SERVICE_RATE = 16

# V2.1: Higher traffic range
MIN_ARRIVAL_RATE = 4
MAX_ARRIVAL_RATE = 12

INTERFERENCE_PENALTY = 2.5

CHANNEL_QUALITY_MIN = 0.6
CHANNEL_QUALITY_MAX = 1.4


# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("dataset_v2_1", exist_ok=True)


# ============================================================
# 3. DATA STORAGE
# ============================================================

dataset = []

scenario_topologies = []

scenario_positions = []


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def create_network():

    """
    Create random cell positions and graph topology.
    """

    positions = np.random.uniform(
        0,
        AREA_SIZE,
        size=(NUM_CELLS, 2)
    )

    adjacency = np.zeros(
        (NUM_CELLS, NUM_CELLS),
        dtype=int
    )

    distances = np.zeros(
        (NUM_CELLS, NUM_CELLS)
    )

    for i in range(NUM_CELLS):

        for j in range(i + 1, NUM_CELLS):

            distance = np.linalg.norm(
                positions[i] - positions[j]
            )

            distances[i, j] = distance
            distances[j, i] = distance

            if distance <= NEIGHBOR_DISTANCE:

                adjacency[i, j] = 1
                adjacency[j, i] = 1

    return positions, adjacency, distances


# ------------------------------------------------------------


def create_base_traffic():

    """
    Give every cell a different base traffic level.
    """

    return np.random.uniform(
        MIN_ARRIVAL_RATE,
        MAX_ARRIVAL_RATE,
        NUM_CELLS
    )


# ------------------------------------------------------------


def create_cell_phases():

    """
    Different cells get different traffic phases.
    """

    return np.random.uniform(
        0,
        2 * np.pi,
        NUM_CELLS
    )


# ------------------------------------------------------------


def traffic_multiplier(
    t,
    cell_id,
    phases,
    scenario_burst_start,
    burst_cells
):

    """
    Generate dynamic traffic.

    Components:

    1. Smooth temporal variation
    2. Cell-specific variation
    3. Temporary traffic bursts
    """

    phase = phases[cell_id]

    # --------------------------------------------------------
    # Smooth traffic variation
    # --------------------------------------------------------

    daily_pattern = (
        1
        + 0.45
        * np.sin(
            2 * np.pi * t / 100
            + phase
        )
    )

    # --------------------------------------------------------
    # Secondary traffic variation
    # --------------------------------------------------------

    secondary_pattern = (
        1
        + 0.20
        * np.sin(
            2 * np.pi * t / 45
            + phase / 2
        )
    )

    multiplier = (
        daily_pattern
        * secondary_pattern
    )

    # --------------------------------------------------------
    # Traffic burst
    # --------------------------------------------------------

    if (
        scenario_burst_start
        <= t
        <
        scenario_burst_start + 30
    ):

        if cell_id in burst_cells:

            multiplier *= 3.0

    return max(
        multiplier,
        0.1
    )


# ------------------------------------------------------------


def allocate_channels(
    previous_channels,
    queue_lengths,
    channel_quality,
    adjacency
):

    """
    Dynamic channel allocation used during simulation.

    IMPORTANT:

    This is NOT the final intelligent allocation
    algorithm of our research.

    It is used to generate realistic training data.

    Cells with larger queues have a higher probability
    of changing channels.
    """

    channels = previous_channels.copy()

    for i in range(NUM_CELLS):

        # ----------------------------------------------------
        # Probability of changing channel
        # ----------------------------------------------------

        queue_factor = (
            queue_lengths[i]
            / QUEUE_CAPACITY
        )

        change_probability = (
            0.10
            + 0.40 * queue_factor
        )

        if np.random.random() < change_probability:

            channel_scores = np.zeros(
                NUM_CHANNELS
            )

            # ------------------------------------------------
            # Evaluate every possible channel
            # ------------------------------------------------

            for c in range(NUM_CHANNELS):

                interference_count = 0

                for j in range(NUM_CELLS):

                    if i != j:

                        if adjacency[i, j] == 1:

                            if channels[j] == c:

                                interference_count += 1

                # --------------------------------------------
                # Channel quality
                # --------------------------------------------

                quality_score = (
                    channel_quality[i, c]
                )

                # --------------------------------------------
                # Interference penalty
                # --------------------------------------------

                interference_score = (
                    INTERFERENCE_PENALTY
                    * interference_count
                )

                channel_scores[c] = (
                    quality_score
                    - interference_score
                )

            # ------------------------------------------------
            # Small random component
            # ------------------------------------------------

            noise = np.random.normal(
                0,
                0.15,
                NUM_CHANNELS
            )

            channel_scores += noise

            # ------------------------------------------------
            # Select best channel
            # ------------------------------------------------

            channels[i] = np.argmax(
                channel_scores
            )

    return channels


# ------------------------------------------------------------


def calculate_interference(
    channels,
    adjacency
):

    """
    Count neighboring cells using the same channel.
    """

    interference = np.zeros(
        NUM_CELLS,
        dtype=int
    )

    for i in range(NUM_CELLS):

        for j in range(NUM_CELLS):

            if i != j:

                if adjacency[i, j] == 1:

                    if channels[i] == channels[j]:

                        interference[i] += 1

    return interference


# ------------------------------------------------------------


def calculate_service_capacity(
    channels,
    interference,
    channel_quality
):

    """
    Calculate service capacity.

    Service capacity depends on:

    - Base service rate
    - Channel quality
    - Interference
    """

    quality = np.array([
        channel_quality[
            i,
            channels[i]
        ]
        for i in range(NUM_CELLS)
    ])

    # --------------------------------------------------------
    # Base service capacity
    # --------------------------------------------------------

    service_capacity = (
        BASE_SERVICE_RATE
        * quality
    )

    # --------------------------------------------------------
    # Interference reduces capacity
    # --------------------------------------------------------

    service_capacity -= (
        INTERFERENCE_PENALTY
        * interference
    )

    # --------------------------------------------------------
    # Minimum service capacity
    # --------------------------------------------------------

    service_capacity = np.maximum(
        service_capacity,
        1
    )

    return service_capacity


# ============================================================
# 5. MAIN SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING DATASET V2.1")
print("=" * 70)


for scenario in range(NUM_SCENARIOS):

    # --------------------------------------------------------
    # CREATE NETWORK
    # --------------------------------------------------------

    positions, adjacency, distances = (
        create_network()
    )

    scenario_positions.append(
        positions
    )

    scenario_topologies.append(
        adjacency
    )


    # --------------------------------------------------------
    # CREATE TRAFFIC PARAMETERS
    # --------------------------------------------------------

    base_arrival_rates = (
        create_base_traffic()
    )

    phases = (
        create_cell_phases()
    )


    # --------------------------------------------------------
    # INITIAL QUEUES
    # --------------------------------------------------------

    queues = np.random.randint(
        0,
        10,
        NUM_CELLS
    )


    # --------------------------------------------------------
    # INITIAL CHANNELS
    # --------------------------------------------------------

    channels = np.random.randint(
        0,
        NUM_CHANNELS,
        NUM_CELLS
    )


    # --------------------------------------------------------
    # INITIAL CHANNEL QUALITY
    # --------------------------------------------------------

    channel_quality = np.random.uniform(
        CHANNEL_QUALITY_MIN,
        CHANNEL_QUALITY_MAX,
        size=(
            NUM_CELLS,
            NUM_CHANNELS
        )
    )


    # --------------------------------------------------------
    # TRAFFIC BURST PARAMETERS
    # --------------------------------------------------------

    burst_start = np.random.randint(
        30,
        160
    )

    burst_cells = np.random.choice(
        NUM_CELLS,
        size=6,
        replace=False
    )


    # ========================================================
    # TIME LOOP
    # ========================================================

    for t in range(TIME_STEPS):


        # ----------------------------------------------------
        # STORE CURRENT QUEUE
        # ----------------------------------------------------

        queue_before = queues.copy()


        # ----------------------------------------------------
        # UPDATE CHANNEL QUALITY
        # ----------------------------------------------------

        channel_quality += np.random.normal(
            0,
            0.03,
            size=(
                NUM_CELLS,
                NUM_CHANNELS
            )
        )

        channel_quality = np.clip(
            channel_quality,
            CHANNEL_QUALITY_MIN,
            CHANNEL_QUALITY_MAX
        )


        # ----------------------------------------------------
        # CALCULATE ARRIVAL RATE
        # ----------------------------------------------------

        arrival_rates = np.zeros(
            NUM_CELLS
        )

        for i in range(NUM_CELLS):

            multiplier = traffic_multiplier(
                t,
                i,
                phases,
                burst_start,
                burst_cells
            )

            arrival_rates[i] = (
                base_arrival_rates[i]
                * multiplier
            )


        # ----------------------------------------------------
        # GENERATE PACKETS
        # ----------------------------------------------------

        arrivals = np.random.poisson(
            arrival_rates
        )


        # ----------------------------------------------------
        # QUEUE AFTER ARRIVALS
        # ----------------------------------------------------

        queue_with_arrivals = (
            queues
            + arrivals
        )


        # ----------------------------------------------------
        # DYNAMIC CHANNEL ALLOCATION
        # ----------------------------------------------------

        channels = allocate_channels(
            channels,
            queue_with_arrivals,
            channel_quality,
            adjacency
        )


        # ----------------------------------------------------
        # CALCULATE INTERFERENCE
        # ----------------------------------------------------

        interference = (
            calculate_interference(
                channels,
                adjacency
            )
        )


        # ----------------------------------------------------
        # CALCULATE SERVICE CAPACITY
        # ----------------------------------------------------

        service_capacity = (
            calculate_service_capacity(
                channels,
                interference,
                channel_quality
            )
        )


        # ----------------------------------------------------
        # PACKETS SERVED
        # ----------------------------------------------------

        packets_served = np.minimum(
            queue_with_arrivals,
            np.floor(
                service_capacity
            )
        )

        packets_served = (
            packets_served.astype(int)
        )


        # ----------------------------------------------------
        # QUEUE AFTER SERVICE
        # ----------------------------------------------------

        queue_after_service = (
            queue_with_arrivals
            - packets_served
        )


        # ----------------------------------------------------
        # PACKET DROPS
        # ----------------------------------------------------

        packet_drops = np.maximum(
            queue_after_service
            - QUEUE_CAPACITY,
            0
        )


        # ----------------------------------------------------
        # FINAL QUEUE
        # ----------------------------------------------------

        queues = np.minimum(
            queue_after_service,
            QUEUE_CAPACITY
        )


        # ----------------------------------------------------
        # QUEUE UTILIZATION
        # ----------------------------------------------------

        utilization = np.divide(
            arrival_rates,
            service_capacity,
            out=np.zeros(
                NUM_CELLS
            ),
            where=service_capacity > 0
        )


        # ----------------------------------------------------
        # ESTIMATED DELAY
        # ----------------------------------------------------

        estimated_delay = np.divide(
            queues,
            arrival_rates,
            out=np.zeros(
                NUM_CELLS
            ),
            where=arrival_rates > 0
        )


        # ====================================================
        # SAVE EACH CELL
        # ====================================================

        for i in range(NUM_CELLS):

            dataset.append({

                "scenario":
                    scenario,

                "time":
                    t,

                "cell_id":
                    i,

                "x_position":
                    positions[i, 0],

                "y_position":
                    positions[i, 1],

                "queue_before":
                    queue_before[i],

                "queue_after":
                    queues[i],

                "base_arrival_rate":
                    base_arrival_rates[i],

                "arrival_rate":
                    arrival_rates[i],

                "arrivals":
                    arrivals[i],

                "channel":
                    channels[i],

                "channel_quality":
                    channel_quality[
                        i,
                        channels[i]
                    ],

                "interference":
                    interference[i],

                "service_capacity":
                    service_capacity[i],

                "packets_served":
                    packets_served[i],

                "utilization":
                    utilization[i],

                "estimated_delay":
                    estimated_delay[i],

                "packet_drops":
                    packet_drops[i]
            })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (scenario + 1) % 50 == 0
        or scenario == 0
    ):

        print(
            f"Completed scenario "
            f"{scenario + 1}/{NUM_SCENARIOS}"
        )


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    dataset
)


# ============================================================
# 7. CREATE FUTURE QUEUE TARGET
# ============================================================

df["future_queue"] = (
    df
    .groupby(
        ["scenario", "cell_id"]
    )["queue_after"]
    .shift(-1)
)


# ============================================================
# 8. REMOVE FINAL TIMESTEP
# ============================================================

df = df.dropna(
    subset=[
        "future_queue"
    ]
)


df["future_queue"] = (
    df["future_queue"]
    .astype(float)
)


# ============================================================
# 9. SAVE DATASET
# ============================================================

csv_path = (
    "dataset_v2_1/"
    "wireless_queue_dataset_v2_1.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 10. SAVE GRAPH TOPOLOGIES
# ============================================================

adjacency_array = np.array(
    scenario_topologies
)

positions_array = np.array(
    scenario_positions
)


np.save(
    "dataset_v2_1/"
    "adjacency_matrices.npy",
    adjacency_array
)

np.save(
    "dataset_v2_1/"
    "positions.npy",
    positions_array
)


# ============================================================
# 11. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET V2.1 GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nNumber of scenarios: "
    f"{NUM_SCENARIOS}"
)

print(
    f"Cells per scenario: "
    f"{NUM_CELLS}"
)

print(
    f"Channels: "
    f"{NUM_CHANNELS}"
)

print(
    f"Timesteps per scenario: "
    f"{TIME_STEPS}"
)

print(
    f"\nDataset rows: "
    f"{len(df):,}"
)

print(
    f"\nDataset columns: "
    f"{len(df.columns)}"
)

print("\nColumns:")

for column in df.columns:

    print(
        f"  - {column}"
    )


# ============================================================
# 12. CONGESTION ANALYSIS
# ============================================================

congested = df[
    df["utilization"] > 1
]

print("\n" + "=" * 70)
print("CONGESTION")
print("=" * 70)

print(
    f"\nCongested samples: "
    f"{len(congested):,}"
)

print(
    f"Congestion percentage: "
    f"{100 * len(congested) / len(df):.2f}%"
)


# ============================================================
# 13. PACKET DROP ANALYSIS
# ============================================================

total_drops = (
    df["packet_drops"].sum()
)

drop_samples = (
    df["packet_drops"] > 0
).sum()

print("\n" + "=" * 70)
print("PACKET DROPS")
print("=" * 70)

print(
    f"\nTotal packet drops: "
    f"{total_drops:,.0f}"
)

print(
    f"Samples with drops: "
    f"{drop_samples:,}"
)


# ============================================================
# 14. INTERFERENCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("INTERFERENCE")
print("=" * 70)

print(
    df["interference"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 15. CHANNEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL DISTRIBUTION")
print("=" * 70)

print(
    df["channel"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 16. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATIONS")
print("=" * 70)

correlation_columns = [

    "arrival_rate",

    "arrivals",

    "channel_quality",

    "interference",

    "service_capacity",

    "utilization",

    "estimated_delay",

    "queue_after",

    "packet_drops",

    "future_queue"
]

correlation_matrix = (
    df[
        correlation_columns
    ]
    .corr()
    .round(3)
)

print(
    correlation_matrix
)


# ============================================================
# 17. SAVE STATISTICS
# ============================================================

df[
    correlation_columns
].describe().to_csv(
    "dataset_v2_1/"
    "statistics.csv"
)


correlation_matrix.to_csv(
    "dataset_v2_1/"
    "correlations.csv"
)


# ============================================================
# 18. QUEUE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)

for cell_id in range(5):

    cell_data = df[
        (df["scenario"] == 0)
        &
        (df["cell_id"] == cell_id)
    ]

    plt.plot(
        cell_data["time"],
        cell_data["queue_after"],
        label=f"Cell {cell_id}"
    )

plt.xlabel(
    "Time Step"
)

plt.ylabel(
    "Queue Length"
)

plt.title(
    "V2.1 Queue Evolution - Scenario 0"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 19. UTILIZATION VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["utilization"],
    bins=50
)

plt.xlabel(
    "Utilization (λ / μ)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "V2.1 Utilization Distribution"
)

plt.grid()

plt.show()


# ============================================================
# 20. ARRIVAL RATE VS QUEUE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sample = df.sample(
    min(10000, len(df)),
    random_state=42
)

plt.scatter(
    sample["arrival_rate"],
    sample["queue_after"],
    alpha=0.15
)

plt.xlabel(
    "Arrival Rate"
)

plt.ylabel(
    "Queue Length"
)

plt.title(
    "Arrival Rate vs Queue Length"
)

plt.grid()

plt.show()


# ============================================================
# 21. NORMAL VS CONGESTED
# ============================================================

normal = df[
    df["utilization"] <= 1
]

congested = df[
    df["utilization"] > 1
]

print("\n" + "=" * 70)
print("NORMAL VS CONGESTED")
print("=" * 70)

print("\nNORMAL:")

print(
    normal[
        [
            "queue_after",
            "arrival_rate",
            "interference",
            "service_capacity",
            "packet_drops"
        ]
    ]
    .mean()
    .round(3)
)


print("\nCONGESTED:")

print(
    congested[
        [
            "queue_after",
            "arrival_rate",
            "interference",
            "service_capacity",
            "packet_drops"
        ]
    ]
    .mean()
    .round(3)
)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "\ndataset_v2_1/"
)

print(
    "  ├── wireless_queue_dataset_v2_1.csv"
)

print(
    "  ├── adjacency_matrices.npy"
)

print(
    "  ├── positions.npy"
)

print(
    "  ├── statistics.csv"
)

print(
    "  └── correlations.csv"
)

print("\nDataset generation finished successfully.")