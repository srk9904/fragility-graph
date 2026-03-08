# FragilityGraph: Tech Stack & Architecture

## 1. High-Level Architecture
This project relies on a lightweight local backend coupled with a rich, 3D-styled web frontend, entirely eliminating the need for heavy local dependencies like Docker or large relational databases. 

**Core Flow:**
1. Code Change Detected (via VS Code Web Extension or Web File Explorer).
2. AST (Abstract Syntax Tree) parses dependencies.
3. Graph Neural Network calculates fragility score.
4. AWS Bedrock provides human-readable explanations.
5. 3D Web Interface visualizes the blast radius.

## 2. Technology Stack & AWS Free Tier Strategy

### 2.1 Backend (Python / FastAPI)
* **Framework:** FastAPI (ASGI setup, native WebSockets).
* **Execution:** Run locally via `uvicorn`.
* **Database (Graph):** Neo4j AuraDB (Cloud Free Tier) completely replacing any local Docker Neo4j instances.
* **Caching:** Upstash Redis (Cloud Free Tier) replacing local Redis servers.
* **Graph Machine Learning:** PyTorch + PyTorch Geometric (CPU optimized, lightweight).
* **AST Parsing:** Tree-sitter (Python).

### 2.2 Frontend (Vanilla JS / WebGL / CSS3)
* **Styling:** Custom CSS featuring realistic glassmorphism, 3D transformations, and fluid animations.
* **Graph Engine:** Cytoscape.js configured for widespread node distribution, zero overlap, and tactile "drag/drop/click" mechanics.
* **Code Viewer:** Highlight.js or Prism.js integrated natively into the floating modals to render exact lines.

### 2.3 VS Code Extension (TypeScript)
* **Core:** Standard VS Code Extension API.
* **Communication:** WebSockets (`ws`) to synchronize instantly with the FastAPI backend without polling.
* **Decorations:** VS Code TextEditorDecorationType to display the inline fragility scores and highlight at-risk lines natively.

### 2.4 AWS Free Tier Services
To completely eradicate local processing bloat, the following AWS services should be utilized:
* **Amazon Bedrock (Nova Micro/Lite):** Used specifically for generating the "Why is this affected?" hover explanations dynamically and quickly.
* **API Gateway & Lambda (Optional):** If the GNN inference can be bundled thinly, port it to Lambda. Otherwise, local FastAPI suffices.

## 3. Deployment Flow (Single Command)
The system must be built such that a single local terminal command spins up the entire environment:
```bash
# Example unified backend/frontend server
python -m uvicorn app.main:app --reload --port 8000
```
This single instance must serve both the AI backend points AND the static HTML frontend.
