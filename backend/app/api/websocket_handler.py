import json
from fastapi import WebSocket
from ..services.ml_service import MLService
from ..graph.neo4j_adapter import Neo4jAdapter

ml_service = MLService()
graph_db = Neo4jAdapter()

async def handle_analysis(websocket, payload):
    from ..services.bedrock_service import bedrock_service
    # 1. Update Graph
    graph_db.update_from_ast(payload)
    
    # 2. Get Predictions (Mocked for now)
    results = ml_service.predict(payload)
    
    # 3. Enrich with AI Explanations for high-risk nodes (score > 0.7)
    for res in results:
        if res.get("score", 0) > 0.7:
            explanation = await bedrock_service.get_fragility_explanation(
                node_name=res["name"],
                impact_count=res.get("impact_count", 5),
                dependencies=res.get("dependencies", ["auth_module", "api_router"])
            )
            res["explanation"] = explanation
            res["alert"] = True
    
    # 4. Send Results
    await websocket.send_json({
        "status": "success",
        "results": results
    })

async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event = message.get("event")
            payload = message.get("payload", {})
            
            if event == "ast_update":
                await handle_analysis(websocket, payload)
            else:
                await websocket.send_json({"error": "Unknown event"})
        except Exception as e:
            await websocket.send_json({"error": str(e)})
            break
