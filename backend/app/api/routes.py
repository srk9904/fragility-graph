"""
API routes for FragilityGraph.

Endpoints:
  GET  /api/v1/health           – healthcheck
  GET  /api/v1/file_tree        – nested project file tree
  GET  /api/v1/line_risks       – per-line risk annotations for a file
  GET  /api/v1/graph_data       – all graph nodes and edges
  POST /api/v1/analyze          – trigger analysis of a single file
  POST /api/v1/analyze_focused  – file-focused analysis with summary
  POST /api/v1/impact_analysis  – change impact analysis
  POST /api/v1/explain_node     – AI explanation for a single node
"""
import os
import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Body
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import FileNode, GraphData, GraphNode, GraphEdge, LineRiskResponse, LineRisk
from app.services import line_analyzer, ml_service, bedrock_service
from app.graph.neo4j_adapter import Neo4jAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Directories to ignore when building the file tree
IGNORED_DIRS = {
    "node_modules", "venv", ".venv", ".git", "__pycache__",
    ".vscode", ".idea", "dist", "build", ".mypy_cache", ".pytest_cache",
    "FinalDocs", ".gemini",
}


# ── Request Models ─────────────────────────────────────────
class ImpactRequest(BaseModel):
    file_path: str
    change_description: str


class ExplainRequest(BaseModel):
    node_id: str
    label: str
    file_path: str
    fragility: float = 0.0


# ── Health ─────────────────────────────────────────────────
@router.get("/health")
def health_check():
    return {"status": "ok"}


# ── File tree ──────────────────────────────────────────────
@router.get("/file_tree")
def get_file_tree(root: Optional[str] = None):
    """Return a nested JSON tree of the project directory."""
    scan_root = root or settings.PROJECT_ROOT
    if not os.path.isdir(scan_root):
        raise HTTPException(status_code=400, detail=f"Not a directory: {scan_root}")
    tree = _build_tree(scan_root)
    return tree


def _build_tree(path: str) -> dict:
    name = os.path.basename(path)
    if os.path.isfile(path):
        return {"name": name, "path": path.replace("\\", "/"), "type": "file"}

    children = []
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        entries = []

    for entry in entries:
        if entry in IGNORED_DIRS or entry.startswith("."):
            continue
        full = os.path.join(path, entry)
        children.append(_build_tree(full))

    return {
        "name": name,
        "path": path.replace("\\", "/"),
        "type": "directory",
        "children": children,
    }


# ── Line risks ─────────────────────────────────────────────
@router.get("/line_risks", response_model=LineRiskResponse)
def get_line_risks(file_path: str = Query(..., description="Absolute path to file")):
    """Parse a file and return per-line risk annotations."""
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    adapter = Neo4jAdapter.get_instance()
    fragile_map = {}
    for node in adapter.get_all_nodes():
        if node.get("fragility", 0) > 0:
            fragile_map[node.get("label", "")] = node["fragility"]

    risks = line_analyzer.compute_line_risks(content, file_path, fragile_map)

    return LineRiskResponse(
        file_path=file_path,
        content=content,
        lines=[LineRisk(**r) for r in risks],
    )


# ── Graph data ─────────────────────────────────────────────
@router.get("/graph_data", response_model=GraphData)
def get_graph_data():
    """Return the full dependency graph for the frontend visualisation."""
    adapter = Neo4jAdapter.get_instance()
    raw_nodes = adapter.get_all_nodes()
    raw_edges = adapter.get_all_edges()

    nodes = [
        GraphNode(
            id=n.get("id", ""),
            label=n.get("label", ""),
            type=n.get("type", "function"),
            file_path=n.get("file_path", ""),
            line_number=n.get("line_number", 0),
            fragility=n.get("fragility", 0.0),
        )
        for n in raw_nodes
    ]
    edges = [
        GraphEdge(
            source=e.get("source", ""),
            target=e.get("target", ""),
            relationship=e.get("relationship", "CALLS"),
        )
        for e in raw_edges
    ]

    return GraphData(nodes=nodes, edges=edges)


# ── Analyze a single file ──────────────────────────────────
@router.post("/analyze")
def analyze_file(file_path: str = Query(...)):
    """Parse a Python file, update the graph, compute fragility, return results."""
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    nodes, edges = line_analyzer.extract_graph_elements(content, file_path)

    adapter = Neo4jAdapter.get_instance()
    adapter.bulk_upsert(nodes, edges)

    scores = ml_service.compute_fragility_scores(nodes, edges)
    for node in nodes:
        score = scores.get(node["id"], 0.0)
        node["fragility"] = score
        adapter.update_fragility_score(node["id"], score)

    fragile_map = {n["label"]: n["fragility"] for n in nodes if n["fragility"] > 0}
    line_risks = line_analyzer.compute_line_risks(content, file_path, fragile_map)

    return {
        "file_path": file_path,
        "nodes": nodes,
        "edges": edges,
        "line_risks": line_risks,
    }


# ── Analyze focused (file-scoped graph + AI summary) ──────
@router.post("/analyze_focused")
def analyze_focused(file_path: str = Query(...)):
    """
    Focused analysis: analyse one file, compute fragility,
    and return graph + line risks + AI file summary.
    """
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Extract graph elements for this file
    nodes, edges = line_analyzer.extract_graph_elements(content, file_path)

    # Push to Neo4j
    adapter = Neo4jAdapter.get_instance()
    adapter.bulk_upsert(nodes, edges)

    # Compute fragility scores
    scores = ml_service.compute_fragility_scores(nodes, edges)
    max_fragility = 0.0
    for node in nodes:
        score = scores.get(node["id"], 0.0)
        node["fragility"] = score
        max_fragility = max(max_fragility, score)
        adapter.update_fragility_score(node["id"], score)

    # Line risks
    fragile_map = {n["label"]: n["fragility"] for n in nodes if n["fragility"] > 0}
    line_risks = line_analyzer.compute_line_risks(content, file_path, fragile_map)

    # AI file summary
    summary = bedrock_service.summarize_file(
        file_path=file_path,
        content=content,
        node_count=len(nodes),
        edge_count=len(edges),
        max_fragility=max_fragility,
    )

    return {
        "file_path": file_path,
        "nodes": nodes,
        "edges": edges,
        "line_risks": line_risks,
        "summary": summary,
        "max_fragility": max_fragility,
    }


# ── Change impact analysis ────────────────────────────────
@router.post("/impact_analysis")
def impact_analysis(req: ImpactRequest):
    """
    Given a file and a change description, identify which functions
    would be affected by the proposed change.
    """
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(req.file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Get existing definitions
    analysis = line_analyzer.analyse_file(content, req.file_path)
    function_names = [d["name"] for d in analysis["definitions"]]

    # Ask Bedrock for impact analysis
    affected = bedrock_service.analyze_change_impact(
        file_path=req.file_path,
        content=content,
        change_description=req.change_description,
        function_names=function_names,
    )

    # Build affected node IDs
    affected_ids = [f"{req.file_path}::{name}" for name in affected]

    return {
        "file_path": req.file_path,
        "change_description": req.change_description,
        "affected_functions": affected,
        "affected_node_ids": affected_ids,
        "total_functions": len(function_names),
    }


# ── Single node AI explanation ─────────────────────────────
@router.post("/explain_node")
def explain_node(req: ExplainRequest):
    """Get an AI explanation for a specific graph node."""
    code_snippet = ""
    if os.path.isfile(req.file_path):
        with open(req.file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Find the function/class definition
        analysis = line_analyzer.analyse_file(content, req.file_path)
        for defn in analysis["definitions"]:
            if defn["name"] == req.label:
                lines = content.splitlines()
                start = max(0, defn["line_start"] - 1)
                end = min(len(lines), defn["line_end"])
                code_snippet = "\n".join(lines[start:end])
                break

    # Get dependencies from graph
    adapter = Neo4jAdapter.get_instance()
    all_edges = adapter.get_all_edges()
    dep_names = [
        e.get("target", "").split("::")[-1]
        for e in all_edges
        if e.get("source") == req.node_id
    ]

    explanation = bedrock_service.explain_fragility(
        function_name=req.label,
        code_snippet=code_snippet if code_snippet else f"# {req.label}",
        fragility_score=req.fragility,
        dependencies=dep_names,
    )

    return {
        "node_id": req.node_id,
        "explanation": explanation,
        "dependencies": dep_names,
    }
