import logging
import hashlib
import random
import re

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model_loaded = False
        logger.info("Initializing ML Service (Enterprise Pulse Engine v3)")
        self.common_deps = [
            "auth_provider.py", "db_adapter.py", "api_v1.js", "session_store.ts",
            "validation_utils.py", "logging_relay.py", "config_loader.py",
            "payment_gateway.py", "notif_service.py", "user_repo.py", "cache_layer.py"
        ]

    def _get_seed(self, text):
        """Generates a consistent seed from a string."""
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def predict(self, payload):
        """
        Calculates fragility and runs a pulse simulation to determine blast radius.
        Supports 'Full File' and 'Intent/Change' modes.
        """
        file_path = payload.get("file_path", "unknown_module.py")
        intent = payload.get("intent", "")
        project = payload.get("project", "default-repo")
        mode = payload.get("mode", "file") # "file" or "intent"

        seed_text = file_path if mode == "file" else intent
        seed = self._get_seed(seed_text + project)
        random.seed(seed)

        # Markdown exception
        is_doc = file_path.lower().endswith(('.md', '.txt', '.pdf', '.docx')) and mode == "file"
        
        nodes = payload.get("nodes", [])
        if not nodes:
            name = file_path.split('/')[-1] if mode == "file" else "ProposedChange"
            nodes = [{"name": name}]

        results = []
        high_risk_keywords = ["auth", "login", "password", "db", "write", "payment", "execute", "token", "core", "kernel", "config", "hop_length", "sample_rate"]
        
        for node in nodes:
            name = node["name"].lower()
            complexity = (seed % 15) + 5
            
            if is_doc:
                score = 0.0
                impact_report = {"total_impacted": 0, "affected_nodes": [], "elements": [{"data": {"id": node["name"], "label": node["name"], "score": 0.0, "type": "root"}}]}
            else:
                base_score = 0.3 + (complexity * 0.02)
                
                # Check keywords in both node name and intent
                combined_text = (name + " " + intent).lower()
                if any(kw in combined_text for kw in high_risk_keywords):
                    base_score += 0.35
                
                # Intent-specific logic: detect variable changes
                if mode == "intent" and " -> " in intent:
                    base_score += 0.2 # Changing values increases structural risk in our simulation
                
                jitter = (seed % 20) / 100.0
                score = min(0.99, base_score + jitter)
                impact_report = self._simulate_pulse(node, score, seed)
            
            # Generate mock code
            mock_code = self._generate_mock_code(node["name"], score, intent, mode)

            results.append({
                "name": node["name"],
                "score": float(f"{score:.2f}"),
                "complexity": complexity if not is_doc else 0,
                "impact_count": impact_report["total_impacted"],
                "blast_radius": impact_report["affected_nodes"],
                "graph_elements": impact_report["elements"],
                "mock_code": mock_code,
                "dependencies": node.get("calls", ["main_router", "config_service", "logger"]) if not is_doc else []
            })
            
        return results

    def _simulate_pulse(self, start_node, initial_strength, seed):
        """BFS with decay to simulate risk propagation."""
        random.seed(seed)
        start_name = start_node["name"]
        elements = [
            {"data": {"id": start_name, "label": start_name, "score": initial_strength, "type": "root"}}
        ]
        
        visited = {start_name}
        affected = []
        queue = [(start_name, initial_strength, 0)]
        decay_factor = 0.75
        
        max_nodes = 12
        node_count = 1
        
        session_deps = random.sample(self.common_deps, min(len(self.common_deps), random.randint(5, 8)))

        while queue and node_count < max_nodes:
            current_name, strength, depth = queue.pop(0)
            if strength < 0.1 or depth > 3: continue
            
            num_neighbors = min(random.randint(2, 3), len(session_deps))
            for i in range(num_neighbors):
                neighbor_name = session_deps.pop() if session_deps else f"SubModule_{random.randint(10,99)}.py"
                if neighbor_name not in visited:
                    visited.add(neighbor_name)
                    weight = round(strength * (decay_factor - (random.random() * 0.1)), 2)
                    
                    elements.append({"data": {"id": neighbor_name, "label": neighbor_name, "score": weight, "type": "leaf"}})
                    elements.append({"data": {
                        "source": current_name, 
                        "target": neighbor_name, 
                        "weight": f"{int(weight*100)}%", 
                        "val": weight
                    }})
                    
                    affected.append({"node": neighbor_name, "strength": weight})
                    queue.append((neighbor_name, weight, depth + 1))
                    node_count += 1
                    if node_count >= max_nodes: break
            if node_count >= max_nodes: break

        return {
            "total_impacted": len(affected) + random.randint(2, 6),
            "affected_nodes": [a["node"] for a in affected[:8]],
            "elements": elements
        }

    def _generate_mock_code(self, file_name, score, intent, mode):
        """Generates mock source code with risk-tagged lines."""
        if file_name.endswith('.md') and mode == "file":
            return [
                {"line": "# Documentation", "risk": "none"},
                {"line": "This is a markdown file with zero structural risk.", "risk": "none"}
            ]
            
        if mode == "intent":
            # Show the change in the code
            change_line = intent
            if " -> " in intent:
                parts = intent.split(" -> ")
                change_line = f"- {parts[0]}\n+ {parts[1]}"
            
            return [
                {"line": f"# Intent: {intent}", "risk": "none"},
                {"line": f"def apply_change():", "risk": "low"},
                {"line": f"    # User Modification detected", "risk": "none"},
                {"line": f"    {change_line}", "risk": "dark-red" if score > 0.8 else "red"},
                {"line": f"    refresh_system_state()", "risk": "medium"},
                {"line": f"    return True", "risk": "none"}
            ]

        # Default full file mock
        return [
            {"line": f"import logging", "risk": "none"},
            {"line": f"from flask import request", "risk": "none"},
            {"line": "", "risk": "none"},
            {"line": f"def process_execution(data: dict):", "risk": "low" if score < 0.5 else "medium"},
            {"line": f"    # [CRITICAL PATH: FRAGILITY {score}]", "risk": "high" if score > 0.7 else "medium"},
            {"line": f"    db_result = await db_write_secure(data)", "risk": "dark-red" if score > 0.85 else ("red" if score > 0.7 else "yellow")},
            {"line": f"    broadcast_to_dependents(db_result)", "risk": "red" if score > 0.6 else "yellow"},
            {"line": f"    return {'{'} 'status': 'success' {'}'}", "risk": "none"}
        ]

ml_service = MLService()
