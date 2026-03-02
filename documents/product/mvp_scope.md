# MVP Scope & Demo Definition

## 1. Minimal Viable Product Scope
The MVP will focus on a high-fidelity demonstration for a **Python** repository.

### Must-Have Features:
- **Language Support:** Python (Function-level analysis).
- **Core Visualization:** Marginal Heatmap (Red/Yellow/Green) in VS Code.
- **ML Engine:** GraphSAGE model providing real-time scores.
- **Explainability:** Bedrock-powered text explanations for "High Risk" scores.
- **Cloud Presence:** Web Explorer (MVP Link) showing a global graph view.

### Excluded for MVP (Post-Hackathon):
- Full support for Polyglot (C++, Go, etc.).
- Deep GitHub App integration (PR comments).
- Automated self-healing code suggestions.

## 2. "The Winning Demo" Scenario
1. **The Setup:** Open a Python repo with a complex dependency structure (e.g., a shared `utils.py` or `auth.py`).
2. **The Change:** Modify a highly-centered "Core Function" that many others call.
3. **The Result:** 
    - Within < 1 second, the VS Code margin turns **Bright Red**.
    - A sidebar panel opens showing: *"Fragility: 0.95. Impact: High. 15 dependent functions in 'payout_service' may be affected. AI Insight: This is a core validator; proceed with caution."*
4. **The Comparison:** Modify a "Leaf Function" (no callers). The margin remains **Green**.

## 3. Success Metrics
- **Performance:** End-to-end latency < 1.5 seconds.
- **Accuracy:** Predicted fragility correlates with structural centrality.
- **Budget:** Total infrastructure cost remains < $300 (utilizing credits).
