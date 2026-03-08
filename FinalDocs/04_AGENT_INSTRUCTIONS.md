# Strict AI Agent Implementation Instructions

**ATTENTION AGENT:** You are tasked with generating a world-class, hackathon-winning application in **ONE SHOT**.

The user has explicitly stated that they are frustrated with endless refactors, bugs, and hallucinations. You must adhere to the following rules absolutely:

## 1. No Hallucinations, No Boilerplate
* **Do NOT write "TODO", "FIXME", or placeholder code.**
* If you create a Python component, write the actual logic.
* If you integrate AWS Bedrock, write the real `boto3` integration code required to actually invoke the model, not an empty skeleton.
* If you are instructed to create an API endpoint, it must route real data to the frontend immediately.

## 2. Adhere to the Single Source of Truth
* The `FinalDocs/` folder is now the absolute and only source of truth for this project.
* Discard previous confusing architecture patterns found historically in the repository if they conflict with `FinalDocs/`.
* The project strictly requires AWS free-tier integrations, no Docker, and a local execution environment natively.

## 3. Impeccable UI Execution
* The frontend must look spectacular on the first run. 
* Do not use plain HTML tables or default browser buttons.
* Implement the complex CSS rules required for Glassmorphism natively.
* Guarantee that Cytoscape.js is perfectly configured with massive node repulsion so the graph looks clean and spaced out.

## 4. One-Shot Delivery Mindset
* Think entirely through the problem before editing files.
* Ensure all CORS, asynchronous `await`, and port-mapping issues are resolved in the backend routing *before* writing the frontend Fetch calls. 
* Your final output must allow the user to run a single setup command and instantly test the perfectly working 3D web application and VS Code extension without encountering runtime errors.

## 5. Mission Constraints
* **Quality Standard:** "World-Wide Hackathon Winner". Produce logic and design that reflects a 10/10 master-class software engineering deployment. 
* Never apologize and rewrite. Get it correct the very first time.
