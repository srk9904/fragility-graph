"""
ML Service — Graph-based fragility scoring.

Uses a lightweight heuristic GNN-style approach:
  - Node features: in-degree, out-degree, code complexity proxies
  - Propagation: message-passing over the adjacency to spread risk
  - Output: per-node fragility score in [0, 100]

This avoids the heavy PyTorch Geometric dependency while following
the same GraphSAGE-inspired algorithm described in design.md.
"""
import logging
import math
from typing import List, Dict

logger = logging.getLogger(__name__)


def compute_fragility_scores(
    nodes: List[Dict],
    edges: List[Dict],
    iterations: int = 3,
    damping: float = 0.85,
) -> Dict[str, float]:
    """
    Compute fragility scores for every node using iterative
    message-passing (PageRank-style with structural features).

    Parameters
    ----------
    nodes : list of dicts with at least {"id", "type", "line_number"}
    edges : list of dicts with {"source", "target"}
    iterations : number of propagation rounds
    damping : propagation decay factor

    Returns
    -------
    dict  node_id → fragility score (0–100)
    """
    if not nodes:
        return {}

    node_ids = {n["id"] for n in nodes}

    # ── Build adjacency ────────────────────────────────────
    in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    out_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    neighbours: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for e in edges:
        src, tgt = e["source"], e["target"]
        if src in node_ids and tgt in node_ids:
            out_degree[src] += 1
            in_degree[tgt] += 1
            neighbours[src].append(tgt)
            neighbours[tgt].append(src)

    n = len(node_ids)

    # ── Initial feature vector (coupling proxy) ────────────
    scores: Dict[str, float] = {}
    for nd in nodes:
        nid = nd["id"]
        coupling = (in_degree.get(nid, 0) + out_degree.get(nid, 0)) / max(n, 1)
        type_weight = 1.2 if nd.get("type") == "function" else 1.0
        scores[nid] = coupling * type_weight

    # ── Iterative propagation ──────────────────────────────
    for _ in range(iterations):
        new_scores: Dict[str, float] = {}
        for nid in node_ids:
            neighbour_sum = sum(scores.get(nb, 0) for nb in neighbours[nid])
            degree = len(neighbours[nid]) or 1
            new_scores[nid] = (1 - damping) * scores[nid] + damping * (neighbour_sum / degree)
        scores = new_scores

    # ── Normalise to 0-100 using sigmoid for absolute scaling ──
    # This avoids the problem where the max node always gets 100.
    # sigmoid(x*5)*100 gives a spread: low coupling → ~20-40, high → 70-95
    result = {}
    for nid, raw in scores.items():
        # Sigmoid-based absolute scaling
        normalised = (1 / (1 + math.exp(-raw * 5))) * 100
        # Shift so that 0 coupling → near 0 (not 50)
        normalised = max(0, (normalised - 50) * 2)
        result[nid] = round(min(normalised, 100), 1)

    return result
