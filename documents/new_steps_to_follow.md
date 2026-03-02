# New Steps to Follow (Full-Fledged Implementation)

This document outlines the step-by-step technical progression to build FragilityGraph optimally, scaling from zero to a full-fledged enterprise system.

## Phase 1: Environment Setup & Core Foundations
1. **Initialize the Monorepo:** Set up a monorepo structure (e.g., using Turborepo or simple subdirectories) containing:
   - `backend/` (FastAPI)
   - `extension/` (VS Code)
   - `ml/` (PyTorch training scripts)
2. **Containerize the Infrastructure:**
   - Create a `docker-compose.yml` at the root that spins up Neo4j Community Edition and Redis.
   - Configure Neo4j with the APOC (Awesome Procedures on Cypher) plugin for advanced graph algorithms.
3. **Establish Backend Skeleton:**
   - Initialize a Python FastAPI project.
   - Setup Neo4j official Python driver connection.
   - Setup Redis connection for WebSocket pub/sub and caching.

## Phase 2: Historical Analysis & ML Data Pipeline
1. **Git Miner Module:** Write a Python script to clone repositories, walk through commit histories, and identify "co-change" files.
2. **Static Dependency Extractor:** Build an offline parser (using Python tree-sitter bindings) to extract file, class, and function-level dependencies from a codebase.
3. **Graph Construction:** Merge the co-change data and static dependency data into Neo4j. Define relationships: `CALLS`, `IMPORTS`, `CO_CHANGES_WITH`.
4. **Label Generation:** Identify bug-fix commits (via commit messages like "fix", "resolves") to determine which components broke. Label these nodes as high-fragility in the dataset.

## Phase 3: Machine Learning Engineering
1. **Feature Engineering:** Write py2neo/Gremlin queries to extract node features (degree, PageRank, cyclomatic complexity, historical change frequency).
2. **Model Construction:** Build a PyTorch Geometric (PyG) model implementing GraphSAGE.
3. **Training Loop:** Train the model offline using the extracted dataset. Optimize for precision over recall (to avoid developer alert fatigue).
4. **Model Serialization:** Save the trained PyTorch model (`.pt`) and write a lightweight inference class in the FastAPI backend that loads this model into memory.

## Phase 4: IDE Extension Development
1. **VS Code Extension Scaffold:** Use `yo code` to generate a TypeScript extension.
2. **Local AST Parsing:** 
   - Add `web-tree-sitter` and the WASM bindings for your target languages (Python, JS).
   - Write logic that listens to `vscode.workspace.onDidChangeTextDocument`.
   - Calculate AST diffs on each keystroke to identify added/removed imports or function calls.
3. **WebSocket Client:** Establish an persistent WSS connection to the FastAPI backend.
4. **IDE Visualizations:** 
   - Embed a Webview in the VS Code sidebar.
   - Use D3.js or Cytoscape.js within the Webview to draw the "Blast Radius".
   - Implement VS Code Diagnostics API to draw red/yellow squiggly lines under high-risk code.

## Phase 5: Real-time Integration & Graph Synchrony
1. **WebSocket Handler (Backend):** Create a FastAPI WebSocket endpoint that authenticates the user and listens for graph delta messages.
2. **Real-time Graph Delta:** 
   - When the extension sends "Added call to Function B from Function A", the backend updates the Neo4j graph incrementally.
   - Trigger a sub-graph extraction around the modified nodes.
3. **Real-time Inference:** Pass the extracted sub-graph features to the in-memory PyTorch model. Calculate the new fragility scores.
4. **Pulse Simulation:** Run the Python algorithm to traverse the graph and calculate the blast radius based on the new fragility scores.
5. **Broadcast Result:** Push the results back over the WebSocket to the IDE.

## Phase 6: Enterprise Features & Scale (Future)
1. **Cloud Migration:**
   - Package the FastAPI app into a Docker container.
   - Deploy Neo4j AuraDB (managed) and Redis Cloud.
   - Deploy FastAPI on AWS ECS Fargate or EKS, sitting behind an Application Load Balancer.
2. **Multi-Language Support:** Expand the VS Code `Tree-sitter` parsers to support Java, Go, C++, etc.
3. **Team Dashboards:** Build a Next.js web application for Tech Leads to view repository-wide technical debt and fragility trends over time.
4. **CI/CD Integration:** Build a GitHub Action / GitLab runner that runs the FragilityGraph analysis on Pull Requests and blocks merges if the blast radius is too risky.
5. **Generative AI (Optional):** Integrate AWS Bedrock / Claude to analyze the blast radius graph and generate plain-English explanations for the GitHub PR comments.
