"""
Line-level risk analyser using Tree-sitter.
Parses Python source, identifies function/class definitions and call-sites,
then returns per-line risk annotations.
"""
import logging
import ast
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
) -> List[Dict]:
    """
    Return per-line risk annotations.

    Parameters
    ----------
    source         : raw file content
    known_fragile  : mapping function_name → fragility_score (0-100)

    Returns
    -------
    list of {line_number, risk_score, reason}
    """
    if known_fragile is None:
        known_fragile = {}

    analysis = analyse_file(source, file_path)
    lines = source.splitlines()
    risks: List[Dict] = []

    # Mark call-sites that invoke fragile functions
    for call in analysis["calls"]:
        fn = call["name"].split(".")[-1]  # strip class prefix
        if fn in known_fragile:
            score = known_fragile[fn]
            
            # Generate more unique reasons
            if score > 90:
                reason = f"Calls fatal dependency `{fn}` (score {score:.0f}). High risk of system-wide failure."
            elif score > 70:
                reason = f"Dependency on critical component `{fn}` (score {score:.0f}). Potential structural bottleneck."
            elif score > 40:
                reason = f"Invocation of `{fn}` (score {score:.0f}). Moderate fragility propagation detected."
            else:
                reason = f"Linked to `{fn}` (score {score:.0f}). Minor fragility risk."

            risks.append(
                {
                    "line_number": call["line"],
                    "risk_score": round(score, 1),
                    "reason": reason,
                }
            )

    # Mark definitions that are themselves fragile
    for defn in analysis["definitions"]:
        if defn["name"] in known_fragile:
            score = known_fragile[defn["name"]]
            
            if score > 90:
                reason = f"Fatal {defn['type']} definition `{defn['name']}` (score {score:.0f})."
            elif score > 40:
                reason = f"Critical {defn['type']} `{defn['name']}` (score {score:.0f}). High maintenance complexity."
            else:
                reason = f"Moderate fragility in {defn['type']} `{defn['name']}` (score {score:.0f})."

            risks.append(
                {
                    "line_number": defn["line_start"],
                    "risk_score": round(score, 1),
                    "reason": reason,
                }
            )

    return risks


def extract_graph_elements(source: str, file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract nodes and edges from a Python file for the dependency graph.

    Returns (nodes_list, edges_list)
    """
    analysis = analyse_file(source, file_path)
    nodes: List[Dict] = []
    edges: List[Dict] = []
    defined_names: set = set()

    for defn in analysis["definitions"]:
        node_id = f"{file_path}::{defn['name']}"
        nodes.append(
            {
                "id": node_id,
                "label": defn["name"],
                "type": defn["type"],
                "file_path": file_path,
                "line_number": defn["line_start"],
                "fragility": 0.0,
            }
        )
        defined_names.add(defn["name"])

    # Build call edges between definitions within the same file
    for call in analysis["calls"]:
        callee = call["name"].split(".")[-1]
        if callee in defined_names:
            # Find which definition contains this call line
            caller_id = None
            for defn in analysis["definitions"]:
                if defn["line_start"] <= call["line"] <= defn["line_end"]:
                    caller_id = f"{file_path}::{defn['name']}"
                    break
            if caller_id:
                target_id = f"{file_path}::{callee}"
                if caller_id != target_id:
                    edges.append(
                        {"source": caller_id, "target": target_id, "relationship": "CALLS"}
                    )

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
