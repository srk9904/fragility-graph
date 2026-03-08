# UI / UX Design Specifications

## 1. Core Aesthetic
The application must present itself as a premium, cutting-edge developer tool.
* **Theme:** "Cyberpunk/Glassmorphism" — Deep blacks, charcoal greys, with vibrant neon accents (Cyan, Magenta, Emerald Green) indicating risk levels.
* **Material:** Extensive use of CSS glassmorphism (translucency + heavy background blurs).
* **Animations:** All transitions (hovering, opening modals, graph layout rendering) must be hardware-accelerated and smooth.

## 2. File Explorer (VS Code Style)
* A dedicated side panel matching the VS Code tree view.
* Clickable folders that expand/collapse fluidly.
* Clicking a file instantly triggers the analysis pipeline and loads the 3D Graph.

## 3. The 3D Dependency Graph
The graph engine (Cytoscape.js) must be extensively configured for a perfect layout:
* **Node Sizing:** Nodes must physically appear small on the screen.
* **Dynamic Typography:** Node labels (filenames) must adjust strictly to fit near the node without overlapping other nodes.
* **Spacious Layout:** The `nodeRepulsion` and `idealEdgeLength` properties MUST be extremely high. The graph layout must enforce a massive distance between individual nodes to guarantee absolutely zero overlapping.
* **Interactions:**
    1. **Click & Drag:** The user can intuitively click and drag any node to physically reposition it across the 3D canvas.
    2. **Click / Double Click:** Executing this action on a specific node will trigger the "Code Modal" workflow for that exact file.

## 4. The Code Viewer Modal
When a dependent node is opened:
* The screen gracefully overlays with a dark glass modal containing the file's raw source code.
* **Precision Highlighting:** Only the exact lines of code that are affected by the "root" dependency are visually highlighted in glowing red/orange (based on risk severity).
* **Interactive AI Explanations:**
    * When the user hovers over any highlighted line, a small, sleek pop-up smoothly slides or floats in from the right side of the screen.
    * This pop-up must contain an explanation from Amazon Bedrock detailing precisely *why* this line is fundamentally broken by the dependency change, and *how* the developer can reduce the blast radius.

## 5. VS Code Extension UX
* The extension must integrate natively without disrupting developer flow.
* A status bar or inline decoration immediately shows the "Fragility Score" when code is saved.
* A one-click "Visualize Blast Radius" button instantly opens the standalone Web Dashboard focused directly on the currently edited file.
