# Machine Learning Prediction Baselines: MLP vs. GraphSAGE

## Project Overview
This part of the project focuses on **predicting the future queue length ($Q_{t+1}$)** of 20 base station cells in a wireless network.

The ultimate research goal is **Queue-Theory Guided Dynamic Channel Allocation**. Before allocating channels, we need an accurate lookahead of which cells will face packet congestion. To prove our method works scientifically, we evaluate three prediction models:
1. **Persistence (Baseline 1):** Assumes tomorrow's queue is identical to today's ($Q_{t+1} = Q_t$).
2. **MLP (Baseline 2):** Predicts future queues using only individual cell statistics.
3. **GraphSAGE GNN (Model 3):** Predicts future queues using individual cell statistics **plus neighbor interference information**.

---

## 1. Multi-Layer Perceptron (MLP) Approach

### What is it in plain English?
The MLP acts like a cell with "blinders" on. It only inspects its own internal dashboard (its current queue, arrival rate, service rate, delay, and packet drops) and ignores what neighboring cells are doing.

### How it works:
* **Input:** A flat vector of 13 features describing a single cell at time $t$.
* **Graph Structure:** **None.** It does not look at edges or adjacent cells.
* **Architecture:**
  * Fully connected linear layers (`nn.Linear`) that transform the 13 numbers into 64 hidden representations.
  * Batch Normalization and ReLU activation functions to help the model learn non-linear patterns.
  * A final linear head that outputs a single number: the estimated queue size at $t+1$.
* **Why we built it:**
  * To see how much predictive power comes strictly from a cell's own local telemetry.
  * It provides the baseline score that our Graph Neural Network must beat.

---

## 2. GraphSAGE (GNN) Approach

### What is it in plain English?
In a wireless cellular network, cells do not operate in a vacuum. If Cell A and Cell B are neighbors using the same frequency channel, they interfere with each other, reducing service capacity and causing packet queues to back up. 

GraphSAGE allows each cell to "talk to its neighbors" before making a prediction.

### How it works:
* **Input:** 
  * Node features (the 13 metrics for each of the 20 cells).
  * Edge connectivity (`edge_index`), representing which cells are physical neighbors or within interference distance.
* **Message Passing (Neighborhood Aggregation):**
  1. **Layer 1 (1-Hop):** Each cell gathers the feature vectors of its direct neighbors, takes the average (`aggr="mean"`), combines this summary with its own features, and updates its representation.
  2. **Layer 2 (2-Hop):** The process repeats. Cell A now gets information about the neighbors of its neighbors.
* **Prediction Head:** 
  * Takes the neighborhood-aware embedding from Layer 2 and passes it through an MLP head to output the predicted future queue.
* **Why we built it:**
  * To test our core hypothesis: **Does spatial topology and neighbor interference data improve queue and congestion forecasting?**

---

## Quick Comparison Summary

| Metric / Aspect | Persistence | MLP (Baseline 2) | GraphSAGE GNN (Model 3) |
| :--- | :--- | :--- | :--- |
| **Sees Cell Features?** | Only current queue ($Q_t$) | Yes (all 13 features) | Yes (all 13 features) |
| **Sees Neighbor State?** | No | No (isolated cells) | **Yes (via graph edges)** |
| **Understands Interference?** | No | Only its own measured impact | **Sees neighbor loads causing it** |
| **Primary Weakness** | Lagging behind surges | Blind to network-wide interference | Slightly higher compute cost |

---

## Current Best Results (Test Set)

* **Baseline 1 (Persistence):**
  * Global MAE: `0.8884` | Congested MAE ($Q \ge 20$): `5.3895`
* **Baseline 2 (MLP):**
  * Global MAE: `0.5874` | Congested MAE ($Q \ge 20$): `3.0150`
* **Model 3 (GraphSAGE):**
  * *(Run `train_gnn.py` to record and compare the reduction in congested error)*

---

## What Comes Next?
Once the GNN predictions are logged and compared against the MLP:
1. Identify high-risk cells predicted to cross the queue stability boundary ($\rho = \lambda / \mu > 1$).
2. Apply queue-theoretic channel allocation rules to dynamically shift spectrum channels from idle cells to congested cells.
3. Validate that reallocating channels reduces overall packet drops and average delay across the network.