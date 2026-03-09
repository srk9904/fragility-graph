# FragilityGraph: Predictive Blast Radius Analysis for Fearless Development

FragilityGraph is an AI-native developer ecosystem designed to eliminate the "fear of refactoring." By combining the topological intelligence of Graph Neural Networks (GNNs) with the advanced reasoning of Amazon Bedrock, FragilityGraph predicts the blast radius of code changes before they are committed, providing real-time risk assessments directly in your IDE and a comprehensive web-based visualizer.

## The Problem: The Hidden Debt of Fragility

In modern, large-scale codebases, a single change to a "fragile" utility function can trigger a cascade of failures across seemingly unrelated modules. Developers often avoid refactoring critical legacy code because they cannot visualize the full impact of their changes. This "fear-driven development" leads to technical debt and brittle systems.

## The Solution: Intelligent Blast Radius Prediction

FragilityGraph solves this by mapping your codebase into a multi-dimensional dependency graph and using AI to score every node's "fragility." 

*   **Proactive Impact Analysis**: See exactly which functions and classes will break BEFORE you save your file.
*   **AI-Powered Explanations**: Amazon Bedrock provides human-readable justifications for risk scores, explaining *why* a node is fragile.
*   **Seamless Integration**: Access insights through a high-performance Web Dashboard or a native VS Code Extension.

## Platform Features

### 1. Web Visualization Dashboard
*   **3D Dependency Mapping**: An interactive graph showing the relationship between functions, classes, and files.
*   **Risk Profile Gauge**: High-level metrics for file-level structural risk and blast radius.
*   **AI File Summary**: Automatic generation of architectural overviews for any selected file.
*   **Interactive Impact Simulation**: Enter a proposed change (e.g., "Add a new parameter to the validate function") and watch the graph light up with affected dependencies.

### 2. VS Code Extension
*   **Real-time Heatmaps**: Visual indicators on the scrollbar and margin showing risky lines.
*   **Hover Insights**: Detailed fragility scores and AI-generated risk explanations accessible directly on function definitions.
*   **One-Click Navigation**: Jump from the IDE directly to the Web Dashboard for deep-dive analysis.

## Technical Architecture

Built on a robust, AWS-native stack, FragilityGraph leverages modern AI patterns for scalability and accuracy.

*   **Intelligence Layer**:
    *   **Amazon Bedrock (Nova Lite)**: Generates product-ready reasoning and impact summaries.
    *   **GraphSAGE (GNN)**: A custom PyTorch-based Graph Neural Network that analyzes AST-derived graph topology to calculate fragility scores.
*   **Data Layer**:
    *   **Neo4j**: High-performance graph database storing the directed graph of code dependencies.
    *   **Redis**: Low-latency caching for real-time responsiveness.
*   **Infrastructure**:
    *   **FastAPI**: High-performance backend orchestrating AI services and graph queries.
    *   **AWS S3**: Secure storage for GNN model weights and metadata.
    *   **AWS Lambda & API Gateway**: Secure, scalable entry points for the extension and web frontend.

## Getting Started

### Prerequisites
*   Python 3.9+
*   Neo4j (Local or AuraDB)
*   Redis
*   AWS Account with Bedrock access (Amazon Nova Lite)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-repo/fragility-graph.git
    cd fragility-graph
    ```

2.  **Environment Setup**
    Create a `.env` file in the root directory:
    ```bash
    AWS_REGION=us-east-1
    BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
    AWS_S3_BUCKET_NAME=your-fragility-model-bucket
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password
    ```

3.  **Backend Setup**
    ```bash
    python -m venv venv
    ./venv/Scripts/activate  # On Windows
    pip install -r backend/requirements.txt
    cd backend
    python -m app.main
    ```

4.  **Frontend Setup**
    Simply open `frontend/index.html` in a modern browser (or serve via Live Server).

5.  **VS Code Extension**
    *   Open the `vscode_extension` folder in VS Code.
    *   Press `F5` to start the Extension Development Host.
    *   Or install the provided `.vsix` file.

## Acknowledgments

This project was developed for the **AWS AI for Bharat Hackathon**. Special thanks to the AWS team for providing the infrastructure and tools (Amazon Bedrock, Nova Models) that made this predictive analysis possible.
