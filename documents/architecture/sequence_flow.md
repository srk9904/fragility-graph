# System Sequence Flow

```mermaid
sequence_flow
Extension -> Local Tree-sitter: Parse Code Change
Local Tree-sitter -> Extension: AST Delta (Added/Updated Nodes)
Extension -> FastAPI (WebSocket): Send event: "ast_update"
FastAPI -> Neo4j: Update Node properties & Edge types
Neo4j -> FastAPI: Fetch k-hop neighborhood graph
FastAPI -> PyTorch (In-Memory): Run GNN Inference on subgraph
PyTorch -> FastAPI: Return Fragility Score & Alert Nodes
FastAPI -> Amazon Bedrock: Provide context (Node + Blast Radius)
Amazon Bedrock -> FastAPI: Return Human-readable Explanation
FastAPI -> Extension (WebSocket): Send event: "fragility_report"
Extension -> VS Code: Render Heatmap & Sidebar Alerts
```

## Detailed Flow Steps:
1. **Trigger:** User saves a file in VS Code.
2. **Local Processing:** Extension uses `web-tree-sitter` to diff the AST. Only modified symbols are extracted.
3. **Transmission:** Secure WebSocket transmits the delta.
4. **Graph Sync:** Neo4j updates the "Live" state of the repository.
5. **Inference:** The GNN model takes the current graph snapshot to calculate the "Impact" of the change.
6. **AI Insight:** Bedrock clarifies *why* the score is high (e.g., "This function is used by 5 critical payout modules").
7. **Display:** Real-time visual feedback to the developer.
