# 📉 FragilityGraph: Predictive Blast Radius Analysis

**FragilityGraph** is an AI-powered developer tool that identifies "fragile" nodes in your codebase before you commit changes. It combines GNN-based topological analysis with Amazon Bedrock’s reasoning to provide real-time risk assessments.

---

## 🚀 How it Works
1.  **Ingestion:** Scans your local Python codebase and generates an AST (Abstract Syntax Tree).
2.  **Graph Mapping:** Ingests the AST into **Neo4j**, creating a directed graph of function calls and dependencies.
3.  **Risk Prediction:** A **GraphSAGE (GNN)** model analyzes the graph topology to assign a "Fragility Score" to every function.
4.  **AI Explanation:** **Amazon Bedrock (Nova Lite)** generates concise, product-ready explanations for high-risk nodes.
5.  **Extension:** A **VS Code Extension** highlights these risks directly in the editor via heatmaps and hovers.

---

## 🛠️ Tech Stack
-   **Backend:** FastAPI (Python)
-   **Database:** Neo4j (Graph) & Redis (Caching)
-   **AI/ML:** Amazon Bedrock (Nova Lite), Boto3, PyTorch (GraphSAGE)
-   **Storage:** Amazon S3 (Model Weights)
-   **Frontend:** VS Code Extension (TypeScript)

---

## ⚙️ Local Setup

### 1. Environment Configuration
Copy `.env.example` to `.env` and fill in your AWS credentials:
```bash
# Required for Bedrock & S3
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
AWS_S3_BUCKET_NAME=your-bucket-name
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Start Infrastructure (Docker)
Ensure Docker is running, then start the database and cache:
```bash
docker-compose up -d
```

### 5. Run the Backend
```bash
cd backend
python -m app.main
```

---

## 🧪 Triggering the Main Workflow (Demo)
We have provided an end-to-end simulation script that validates the entire pipeline (GNN + Bedrock).

**Run the Demo:**
```bash
.\venv\Scripts\python verification/run_demo.py
```

### Ready-to-Run Test Cases
| Scenario | Input | Expected Output |
| :--- | :--- | :--- |
| **Critical Auth** | `validate_session` | Score > 0.8, Explains cascading login failure |
| **Legacy Payment** | `execute_transaction` | Score > 0.9, Explains high risk of third-party failure |
| **Utility Refactor** | `format_currency` | Score < 0.3, No alert generated |

---

## 📂 Project Structure
-   `backend/`: FastAPI application & AI services.
-   `extension/`: VS Code extension source (TypeScript).
-   `verification/`: Test scripts and demo simulations.
-   `documents/`: Architecture and design artifacts.

---
*Developed for the AWS AI for Bharat Hackathon.*
