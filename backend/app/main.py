import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from .api.websocket_handler import handle_websocket
from .services.config import settings
from .services.ml_service import ml_service
from .services.bedrock_service import bedrock_service

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="FragilityGraph Backend")

# Mount Static Files for DeepPulse Dashboard
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
def read_root():
    return {"message": "FragilityGraph API is active. Visit /dashboard for the Pulse Visualizer."}

@app.post("/api/v1/analyze_mock")
async def analyze_mock(payload: dict):
    """
    Direct endpoint for the Web Dashboard to trigger the full AI/ML pipeline.
    """
    results = ml_service.predict(payload)
    for res in results:
        if res['score'] > 0.4:
            res['explanation'] = await bedrock_service.get_fragility_explanation(
                node_name=res['name'],
                impact_count=res['impact_count'],
                dependencies=res['dependencies']
            )
    return results

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        await handle_websocket(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket communication: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
