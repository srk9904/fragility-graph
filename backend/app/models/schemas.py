from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ── File Tree ──────────────────────────────────────────────
class FileNode(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
    children: Optional[List["FileNode"]] = None


# ── Graph data sent to frontend ────────────────────────────
class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # function | class | module
    file_path: str
    line_number: int
    fragility: float = 0.0


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str  # CALLS | IMPORTS | BELONGS_TO


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ── Line Risk ─────────────────────────────────────────────
class LineRisk(BaseModel):
    line_number: int
    risk_score: float
    reason: str


class LineRiskResponse(BaseModel):
    file_path: str
    content: str
    lines: List[LineRisk]


# ── WebSocket payloads ─────────────────────────────────────
class CodeChangePayload(BaseModel):
    file_path: str
    content: str
    language: str = "python"


class FragilityUpdate(BaseModel):
    file_path: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    line_risks: List[LineRisk]
    explanation: str
