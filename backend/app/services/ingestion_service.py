import os
import ast
import logging
from ..graph.neo4j_adapter import Neo4jAdapter

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.graph_db = Neo4jAdapter()

    def ingest_local_path(self, root_path):
        """
        Recursively walks through a local path and ingests all Python files.
        """
        logger.info(f"Starting ingestion for path: {root_path}")
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    self._ingest_file(full_path, root_path)

    def _ingest_file(self, file_path, root_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            nodes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    nodes.append({
                        "name": node.name,
                        "type": "function",
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno + 1),
                        "calls": self._find_calls(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    nodes.append({
                        "name": node.name,
                        "type": "class",
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno + 1)
                    })
            
            payload = {
                "file_path": os.path.relpath(file_path, root_path),
                "nodes": nodes
            }
            self.graph_db.update_from_ast(payload)
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")

    def _find_calls(self, func_node):
        calls = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        return list(set(calls))
