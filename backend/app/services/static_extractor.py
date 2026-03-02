import ast
import os
import logging

logger = logging.getLogger(__name__)

class StaticExtractor:
    def __init__(self, root_path):
        self.root_path = root_path

    def analyze_file(self, file_path):
        """
        Parses a python file and extracts functions, classes, and calls.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            nodes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract calls within this function
                    calls = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                calls.append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                calls.append(child.func.attr)
                    
                    nodes.append({
                        "name": node.name,
                        "type": "function",
                        "line_start": node.lineno,
                        "complexity": self._calculate_complexity(node),
                        "calls": list(set(calls))
                    })
            return nodes
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return []

    def _calculate_complexity(self, node):
        """Simplified cyclomatic complexity (branch counting)."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def scan_repo(self):
        """Scans the entire repository for python files."""
        all_data = {}
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.root_path)
                    all_data[rel_path] = self.analyze_file(full_path)
        return all_data

# Placeholder for use
extractor = StaticExtractor(os.getcwd())
