# 🚀 Fragility Graph: The Ultimate Demo Script

**Theme**: "Moving from Reactive Debugging to Proactive Architecture."
**Target Duration**: 3-5 Minutes

---

## 🕒 [0:00 - 0:45] The Hook & Project Vision
**Visual**: Show a complex codebase (e.g., the root directory of this project) in a VS Code-like File Tree.

**Narration**: 
> "Every developer knows the fear of changing one line of code and watching the entire system collapse. Modern codebases aren't just collections of files; they are living, breathing webs of invisible dependencies. 
>
> Today, we're introducing **Fragility Graph**. We leverage AI and graph theory to map the 'Blast Radius' of your changes before you even hit save. We don't just track imports; we track *architectural risk*."

---

## 🔍 [0:45 - 1:30] Navigation & Real-Time Mapping
**Action**: Click through the **File Tree**. Select `backend/app/main.py` or a core service.
**Visual**: The graph nodes should swirl and center themselves (Auto-fit).

**Narration**:
> "As I navigate through the project, Fragility Graph builds a real-time, bidirectional map of the project. Notice the **Blue Node**? That's our focus. The **Orange Nodes** are its 'Blast Radius'—files that will break if our focus file fails.
>
> Unlike static analysis tools, our graph is **multidimensional**. It calculates a **Fragility Score** using AI-driven complexity analysis, showing you exactly where the 'thin ice' is in your architecture."

---

## 💻 [1:30 - 2:30] Deep-Dive: Intelligent Code View
**Action**: Click an orange node. Select **View Full Code**.
**Visual**: Show the code viewer with **multi-line block highlights**. Hover over a highlight to show the tooltip.

**Narration**:
> "Let's look deeper. When I open the code for a dependent module, I'm not just seeing code; I'm seeing **Architectural Reasoning**. 
>
> Look at these highlighted blocks. The system isn't just flagging an import; it's identifying the **specific functions and class bodies** that are actively tied to our focused file. Our **Intelligent Multi-Line Highlighting** ensures you see the full context of the risk, with AI-generated tooltips explaining the *type* of dependency—whether it's a direct impact or a deep structural link."

---

## 🔥 [2:30 - 3:30] The "Wow" Moment: Predictive Impact Analysis
**Action**: Go to the **Impact Analysis** box. Type: *"Rename the database driver instance to 'db_client' and change its initialization signature."*
**Visual**: Click **Run Impact Analysis**. Watch the graph pulse red across multiple files.

**Narration**:
> "This is the game-changer. I'm about to make a major structural change. Instead of waiting for CI to fail 10 minutes from now, I ask the **Impact Analysis** engine. 
>
> In seconds, the system simulates the change and propagates the risk through the graph. See that? Almost the entire backend services layer is pulsing. The AI reasoning engine tells me exactly *why*—this singleton is used across 15 different modules. We just saved ourselves two hours of debugging by identifying a 'High Risk' architectural move in real-time."

---

## 🏆 [3:30 - EOF] Closing & Technical Punch
**Visual**: Zoom out to show the full, vibrant graph.

**Narration**:
> "Fragility Graph is built on a stack of AWS Bedrock for reasoning and Neo4j for relationship mapping. We're moving development from 'Guess and Check' to 'See and Know'. 
>
> We make codebases safer, refactoring faster, and developers more confident. Fragility Graph: Predict the blast radius, prevent the fire. Thank you."

---

## 💡 Key Tips for a World-Class Demo:
1.  **Energy**: Speak with authority. You aren't just showing a tool; you're showing the *future*.
2.  **Smooth Transitions**: Practice the click-path so there's no dead air while loading.
3.  **Color Palette**: Ensure your screen-share shows off the **Neon-Glassmorphism** UI styles we built—it looks premium and "world-class" on high-res monitors.
4.  **The "Big Break"**: Use the most complex file you have for the Impact Analysis demo. The more nodes that pulse, the more impressive it is.
