# ML Specification: Fragility Prediction

## 1. Problem Definition
Predict the **structural fragility** of a code element (function/class).
- **Target:** Regression. A score between `0.0` (stable) and `1.0` (extremely fragile/risky).
- **Definition of Fragility:** High probability that a change in this node will cause unexpected regressions in downstream components, or that the node itself is prone to frequent breaking changes due to complex dependencies.

## 2. Model Architecture
- **Type:** GraphSAGE (GNN).
- **Input:** Subgraph surrounding the target node (k-hop neighbors).
- **Node Features:**
    - `degree_centrality`: Number of callers/callees.
    - `cyclomatic_complexity`: Code complexity.
    - `loc`: Lines of Code.
    - `change_frequency`: Historical git frequency.
    - `node_type_enc`: One-hot encoding of (Function, Class, Module).

## 3. Training Data
- **Sources:**
    - Historical Git commit logs (to find "bug-fix" clusters).
    - Synthetic labels: Calculated based on centrality measures and churn metrics for the MVP.
- **Labels (y):** Weighted average of `churn` and `bug_fix_count` for that node.

## 4. Output
- **Primary:** `fragility_score` per node.
- **Secondary:** `blast_radius_nodes` (Nodes most likely to break if this node is modified).
