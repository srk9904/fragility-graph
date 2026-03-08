# FragilityGraph: Complete Requirements & Project Scope

## 1. Project Overview & Vision
FragilityGraph is an AI-powered code dependency and "blast radius" visualization tool designed to win world-wide hackathons. It identifies fragile code dependencies, visually maps them in a smooth 3D-styled user interface, and provides real-time AI reasoning on exactly why specific lines of code are vulnerable.

The ultimate goal is a pristine, one-shot deployable system with zero setup fluff, running entirely locally without draining storage and relying on AWS Free Tier services for heavy lifting.

## 2. Core Functional Requirements

### 2.1 The Web Dashboard (Primary Interface)
* **Smooth 3D UI Design:** The application must have a highly polished, interactive 3D aesthetic. Transitions should be smooth, components should feel tactile, and user feedback must be immediate.
* **File Explorer Component:** A prominent, VS Code-styled file explorer allowing users to navigate their codebase seamlessly. Expanding/collapsing folders and clicking files should trigger the analysis and layout exactly as an IDE would.
* **Interactive Dependency Graph:**
    * **Distribution:** Nodes must be distributed logically with a large distance between them to absolutely prevent any overlapping. 
    * **Node Size & Labels:** Nodes must be small, with labels (filenames) dynamically adjusting to node size without causing clutter.
    * **Interactions:**
        * *Click & Drag:* Freely move and adjust nodes around the screen.
        * *Single/Double Click:* Clicking a node opens a precise code viewer highlighting the affected components within that file.
* **Granular Code Highlighting & Explanations:**
    * If a dependent node is opened, the exact lines of code affected by the root dependency must be highlighted perfectly.
    * **Hover Action:** Hovering over these highlighted lines must trigger a smooth pop-up floating from the right side. This pop-up provides an AI-generated explanation detailing *why* this line is affected and *how to reduce the blast radius*.

### 2.2 VS Code Extension Integration
* **Real-time Monitoring:** The extension sits actively in the editor. As the user makes changes and saves a file, it calculates the fragility score.
* **Score Visibility:** The calculated fragility score must be immediately visible to the developer inside the editor.
* **Seamless Dashboard Navigation:** The extension allows the user to immediately jump/navigate to the Web Dashboard to visualize the blast radius of their recent change, perfectly mirroring the web workflow.

## 3. Strict Development Constraints
* **No Docker:** Do NOT use Docker or containerization. The system must run entirely on local environments seamlessly (e.g., Python `venv`, Node/NPM).
* **Storage Optimization:** Do not use massive local databases or temporary file bloat. Depend on in-memory operations or lightweight cloud alternatives.
* **AWS Free Tier Services:** Leverage AWS Free Tier options aggressively (e.g., lightweight Bedrock models, Lambda, DynamoDB/RDS if required) to keep local overhead at absolute zero.
* **Zero Fluff:** Provide a directly working system without boilerplate, empty "dummy" mocks, or "TODO" implementations. It must be a complete product.
