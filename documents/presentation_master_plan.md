# 🏆 FragilityGraph Presentation Master Plan

To win the **AWS AI for Bharat Hackathon**, we must align with the 4-stage evaluation. This document contains the blueprints for your PPT, Video, and MVP.

---

## 🎨 1. PPT Structure (The Gatekeeper)

**Slide 1: Title & Hook**
- *FragilityGraph: Predicting Software Failures Before They Happen.*
- Impact Statement: "80% of outages are caused by 'innocent' changes to fragile code. We fix that."

**Slide 2: The Problem (The Gap)**
- Large codebases are black boxes.
- Developers don't know the "Blast Radius" of their changes.
- Current tools are reactive (detect bugs after they break).

**Slide 3: The Solution (The Innovation)**
- Real-time GNN-based topological analysis.
- Detects structural fragility and temporal coupling (Git Mining).
- Powered by **Amazon Bedrock** for human-readable risk explanations.

**Slide 4: Technical Architecture (The AWS Edge)**
- **Amazon S3**: Secure storage for GNN model weights.
- **Amazon Bedrock (Nova/Claude)**: The "Contextual Reasoner."
- **Neo4j**: Graph database for infinite relationship traversing.
- **FastAPI**: Real-time WebSocket processing.

**Slide 5: Business Impact**
- Reduction in production MTTR (Mean Time to Repair).
- Cost savings on AWS infrastructure by preventing buggy deployments.
- Improved developer velocity.

---

## 🎬 2. Video Demo Script (The Wow Factor)

**Intro (0:00 - 0:15):**
"Hi, I'm [Name]. Today, I'm showing you FragilityGraph—the future of predictive code analysis."

**DeepPulse Dashboard (0:15 - 0:45):**
*Record your screen at `localhost:8000/dashboard`*
"Here is our DeepPulse Dashboard. We're simulating a change to a critical Auth module. Notice the Pulse simulation starting... instantly, our GNN predicts a 98% risk score."

**AI Reasoning (0:45 - 1:15):**
"But scores aren't enough. Powered by **Amazon Bedrock**, FragilityGraph explains *why* it's risky. It identifies that 13 downstream services depend on this token validation logic, reaching into the database and session stores."

**VS Code Vision (1:15 - 1:45):**
"In the developer's chair, this is a VS Code extension that marks files with heatmaps, preventing the 'Save-and-Pray' workflow."

**Outro (1:45 - 2:00):**
"FragilityGraph: AI-driven safety for modern development. Built on AWS."

---

## 🔗 3. MVP Link (The POC)
Your MVP Link for reviewers should point to the **DeepPulse Dashboard**.

**Setup for Submission:**
1. Ensure the FastAPI server is running.
2. Provide the URL: `http://localhost:8000/dashboard` (or your public IP if deployed).
3. Mention that it leverages the **Amazon Bedrock Nova Lite** model for live reasoning.

---

## 🏁 4. Final Audit (Pre-Submission)
- [x] Run `python verification/run_test_suite.py` (Ensure 3/3 Pass).
- [x] Check `README.md` for clear one-command setup.
- [x] Verify AWS Credentials are set in `.env`.
