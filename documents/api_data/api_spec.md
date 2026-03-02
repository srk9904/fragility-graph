# API Specification

## 1. WebSocket Interface (`/ws/analyze`)

The VS Code extension maintains a persistent connection for real-time updates.

### Message: AST Update (Client -> Server)
Sent whenever a file is saved or a significant pause in typing occurs.
```json
{
  "event": "ast_update",
  "payload": {
    "repo_id": "string",
    "file_path": "string",
    "language": "python|javascript",
    "nodes": [
      {
        "id": "string",
        "type": "function|class|module",
        "name": "string",
        "line_start": 1,
        "line_end": 10,
        "calls": ["string"] 
      }
    ],
    "deleted_nodes": ["string"]
  }
}
```

### Message: Fragility Result (Server -> Client)
Pushed back after processing.
```json
{
  "event": "fragility_report",
  "payload": {
    "file_path": "string",
    "scores": {
      "func_name": 0.82,
      "class_name": 0.45
    },
    "alerts": [
      {
        "node_id": "string",
        "severity": "high|medium|low",
        "blast_radius": ["string"],
        "explanation": "AI-generated text from Bedrock"
      }
    ]
  }
}
```

## 2. REST Endpoints

### POST `/api/v1/ingest`
Initial repository ingestion.
- **Body:** `{ "repo_url": "string", "branch": "string" }`
- **Response:** `{ "job_id": "string", "status": "processing" }`

### GET `/api/v1/status/{job_id}`
Check ingestion status.

### GET `/api/v1/heatmap/{repo_id}`
Fetch global heatmap for the Web Explorer.
