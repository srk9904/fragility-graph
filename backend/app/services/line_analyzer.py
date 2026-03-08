"""
Line-level risk analyser using Tree-sitter.
Parses Python source, identifies function/class definitions and call-sites,
then returns per-line risk annotations.
"""
import logging
import ast
import os
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def analyse_file(source: str, file_path: str = "") -> Dict:
    """
    Parse a Python file and return structured analysis data.

    Returns
    -------
    dict with keys:
        definitions – list of {name, type, line_start, line_end}
        calls       – list of {name, line, col}
        imports     – list of {module, line}
    """
    definitions: List[Dict] = []
    calls: List[Dict] = []
    imports: List[Dict] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.warning("SyntaxError parsing %s: %s", file_path, e)
        return {"definitions": [], "calls": [], "imports": []}

    for node in ast.walk(tree):
        # Functions & methods
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(
                {
                    "name": node.name,
                    "type": "function",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                }
            )
        # Classes
        elif isinstance(node, ast.ClassDef):
            definitions.append(
                {
                    "name": node.name,
                    "type": "class",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                }
            )
        # Call-sites
        elif isinstance(node, ast.Call):
            name = _resolve_call_name(node.func)
            if name:
                calls.append({"name": name, "line": node.lineno, "col": node.col_offset})
        # Imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            imports.append({"module": node.module or "", "line": node.lineno})

    return {"definitions": definitions, "calls": calls, "imports": imports}


def compute_line_risks(
    source: str,
    file_path: str = "",
    known_fragile: Dict[str, float] | None = None,
    focus_path: str = "",
) -> List[Dict]:
    """
    Return per-line risk annotations with intelligent bidirectional matching.
    """
    if known_fragile is None:
        known_fragile = {}

    analysis = analyse_file(source, file_path)
    risks: List[Dict] = []
    
    norm_current = os.path.abspath(file_path).replace("\\", "/") if file_path else ""
    norm_focus = os.path.abspath(focus_path).replace("\\", "/") if focus_path else ""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # 1. Bidirectional Dependency Analysis
    if focus_path and norm_current != norm_focus:
        # A. Get focus file's analysis for cross-referencing
        try:
            with open(focus_path, "r", encoding="utf-8", errors="replace") as f:
                focus_source = f.read()
            focus_analysis = analyse_file(focus_source, focus_path)
        except:
            focus_analysis = {"definitions": [], "calls": [], "imports": []}

        # Case 1: THIS file imports the FOCUS (Dependent)
        found_import = False
        for imp in analysis["imports"]:
            module_path = imp["module"].replace(".", "/")
            target_file = os.path.join(project_root, f"{module_path}.py")
            if not os.path.isfile(target_file):
                target_file = os.path.join(project_root, module_path, "__init__.py")

            if os.path.isfile(target_file) and os.path.abspath(target_file).replace("\\", "/") == norm_focus:
                risks.append({
                    "line_number": imp["line"],
                    "risk_score": 60.0,
                    "reason": f"Direct Impact: This file imports the focused component `{os.path.basename(focus_path)}`."
                })
                found_import = True
        
        # If we import the focus, also check for calls to focus's functions
        if found_import:
            focus_defs = {d["name"] for d in focus_analysis["definitions"]}
            for call in analysis["calls"]:
                name = call["name"].split(".")[-1]
                if name in focus_defs:
                    risks.append({
                        "line_number": call["line"],
                        "risk_score": 75.0,
                        "reason": f"Active Usage: Calls function `{name}` from the focused fragile module."
                    })

        # Case 2: FOCUS imports THIS (Dependency)
        # Highlight our functions that the focus is actually calling
        our_defs = {d["name"]: d for d in analysis["definitions"]}
        for f_call in focus_analysis["calls"]:
            fname = f_call["name"].split(".")[-1]
            if fname in our_defs:
                d = our_defs[fname]
                risks.append({
                    "line_number": d["line_start"],
                    "line_end": d["line_end"],
                    "risk_score": 55.0,
                    "reason": f"Structural Usage: This {d['type']} is being used by the focused component `{os.path.basename(focus_path)}`."
                })

    # 2. Fragility-based highlighting (Internal)
    for call in analysis["calls"]:
        fn = call["name"].split(".")[-1]
        if fn in known_fragile:
            score = known_fragile[fn]
            risks.append({
                "line_number": call["line"],
                "risk_score": round(score, 1),
                "reason": f"Internal Risk: Calls fragile `{fn}` (score {score:.0f}).",
            })

    for defn in analysis["definitions"]:
        if defn["name"] in known_fragile:
            score = known_fragile[defn["name"]]
            risks.append({
                "line_number": defn["line_start"],
                "line_end": defn["line_end"],
                "risk_score": round(score, 1),
                "reason": f"Fragility Source: Critical {defn['type']} definition `{defn['name']}` (score {score:.0f}).",
            })

    # Sort and deduplicate
    unique_risks = {}
    for r in risks:
        ln = r["line_number"]
        if ln not in unique_risks or r["risk_score"] > unique_risks[ln]["risk_score"]:
            unique_risks[ln] = r
            
    return list(unique_risks.values())


def extract_graph_elements(
    source: str, file_path: str, max_depth: int = 2
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build a project-wide BIDIRECTIONAL neighborhood graph (dist <= max_depth).
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    def _normalize(p: str) -> str:
        return os.path.abspath(p).replace("\\", "/")

    norm_root = _normalize(file_path)
    
    # 1. Build the full project-wide import map
    full_graph: Dict[str, List[str]] = {} # source_file -> list of target_files
    all_files = []
    for root, _, files in os.walk(project_root):
        if any(d in root for d in ["venv", ".git", "__pycache__", "node_modules"]):
            continue
        for f in files:
            if f.endswith(".py"):
                all_files.append(os.path.join(root, f))

    for fp in all_files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            analysis = analyse_file(content, fp)
            norm_fp = _normalize(fp)
            targets = []
            for imp in analysis["imports"]:
                m_path = imp["module"].replace(".", "/")
                t_file = os.path.join(project_root, f"{m_path}.py")
                if not os.path.isfile(t_file):
                    t_file = os.path.join(project_root, m_path, "__init__.py")
                if os.path.isfile(t_file):
                    targets.append(_normalize(t_file))
            full_graph[norm_fp] = targets
        except:
            continue

    # 2. Extract bidirectional neighborhood (BFS)
    neighborhood_nodes: set = {norm_root}
    queue: List[Tuple[str, int]] = [(norm_root, 0)]
    visited = {norm_root}
    
    while queue:
        curr, dist = queue.pop(0)
        if dist >= max_depth:
            continue
        
        # Outgoing (Current imports X)
        for target in full_graph.get(curr, []):
            if target not in visited:
                visited.add(target)
                neighborhood_nodes.add(target)
                queue.append((target, dist+1))
        
        # Incoming (X imports Current)
        for source_node, targets in full_graph.items():
            if curr in targets:
                if source_node not in visited:
                    visited.add(source_node)
                    neighborhood_nodes.add(source_node)
                    queue.append((source_node, dist+1))

    # 3. Collect final nodes and edges
    nodes: List[Dict] = []
    edges: List[Dict] = []
    
    for node_id in neighborhood_nodes:
        nodes.append({
            "id": node_id,
            "label": os.path.basename(node_id),
            "type": "file",
            "file_path": node_id,
            "line_number": 1,
            "fragility": 0.0,
        })
        
    for source_node, targets in full_graph.items():
        if source_node in neighborhood_nodes:
            for target in targets:
                if target in neighborhood_nodes:
                    edges.append({
                        "source": source_node,
                        "target": target,
                        "relationship": "IMPORTS",
                    })

    return nodes, edges


# ── Helpers ────────────────────────────────────────────────
def _resolve_call_name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _resolve_call_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    return None
