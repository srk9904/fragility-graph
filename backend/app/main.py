"""
FragilityGraph — main entry point.

Run with:
    cd backend
    python -m app.main
"""
import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.api.routes import router
from app.graph.neo4j_adapter import Neo4jAdapter
from app.services import line_analyzer, ml_service, bedrock_service

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated on_event) ────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup
    logger.info("Starting FragilityGraph …")
    adapter = Neo4jAdapter.get_instance()
    adapter.connect()
    yield
    # Shutdown
    logger.info("Shutting down FragilityGraph …")
    Neo4jAdapter.get_instance().close()


# ── App ────────────────────────────────────────────────────
app = FastAPI(title="FragilityGraph", version="1.0.0", lifespan=lifespan)

# Register REST API routes
app.include_router(router)


# ── WebSocket endpoint (registered BEFORE static mount) ───
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    logger.info("WebSocket client connected")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
                continue

            file_path = payload.get("file_path", "")
            content = payload.get("content", "")

            if not file_path or not content:
                await ws.send_json({"error": "file_path and content required"})
                continue

            try:
                # 1. Extract graph elements
                nodes, edges = line_analyzer.extract_graph_elements(content, file_path)

                # 2. Push to Neo4j
                adapter = Neo4jAdapter.get_instance()
                adapter.bulk_upsert(nodes, edges)

                # 3. ML fragility scoring
                scores = ml_service.compute_fragility_scores(nodes, edges)
                for node in nodes:
                    score = scores.get(node["id"], 0.0)
                    node["fragility"] = score
                    adapter.update_fragility_score(node["id"], score)

                # 4. Line risk analysis
                fragile_map = {n["label"]: n["fragility"] for n in nodes if n["fragility"] > 0}
                line_risks = line_analyzer.compute_line_risks(content, file_path, fragile_map)

                # 5. AI explanation (pick highest fragility node)
                explanation = ""
                if nodes:
                    top_node = max(nodes, key=lambda n: n.get("fragility", 0))
                    if top_node["fragility"] > 0:
                        lines = content.splitlines()
                        start = max(0, top_node["line_number"] - 1)
                        snippet = "\n".join(lines[start: start + 15])
                        dep_names = [
                            e["target"].split("::")[-1]
                            for e in edges
                            if e["source"] == top_node["id"]
                        ]
                        explanation = bedrock_service.explain_fragility(
                            function_name=top_node["label"],
                            code_snippet=snippet,
                            fragility_score=top_node["fragility"],
                            dependencies=dep_names,
                        )

                # 6. Send update back
                update = {
                    "type": "fragility_update",
                    "file_path": file_path,
                    "nodes": nodes,
                    "edges": edges,
                    "line_risks": line_risks,
                    "explanation": explanation,
                }
                await ws.send_json(update)
                await manager.broadcast(update)

            except Exception as proc_err:
                logger.error("Processing error: %s", proc_err, exc_info=True)
                await ws.send_json({"error": str(proc_err)})

    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        manager.disconnect(ws)


# ── Mount frontend LAST (catch-all for static files) ───────
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    logger.info("Frontend mounted from %s", frontend_path)
else:
    logger.warning("Frontend directory not found at %s", frontend_path)


# ── Direct execution ───────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.BACKEND_PORT,
        reload=True,
    )
