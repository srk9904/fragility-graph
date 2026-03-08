"""
Stress-test runner for FragilityGraph.
Assumes the server is already running on http://localhost:8000.
Run:
    python tests/stress_test.py
"""
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import os

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def log(ok, tag, detail=""):
    global PASS, FAIL
    icon = "[PASS]" if ok else "[FAIL]"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {icon}  {tag}  {detail}")


def get(path):
    url = BASE + path
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=15)
    body = resp.read().decode()
    return resp.status, body


def post(path):
    url = BASE + path
    req = urllib.request.Request(url, method="POST", data=b"")
    resp = urllib.request.urlopen(req, timeout=30)
    body = resp.read().decode()
    return resp.status, body


# ── 1. Health ──────────────────────────────────────────────
print("\n[1] Health Check")
try:
    code, body = get("/api/v1/health")
    data = json.loads(body)
    log(code == 200 and data.get("status") == "ok", "GET /api/v1/health", f"HTTP {code}")
except Exception as e:
    log(False, "GET /api/v1/health", str(e))

# ── 2. File Tree ───────────────────────────────────────────
print("\n[2] File Tree")
try:
    code, body = get("/api/v1/file_tree")
    data = json.loads(body)
    has_children = isinstance(data.get("children"), list) and len(data["children"]) > 0
    log(code == 200 and has_children, "GET /api/v1/file_tree", f"HTTP {code}, {len(body)} bytes, {len(data.get('children', []))} root entries")
except Exception as e:
    log(False, "GET /api/v1/file_tree", str(e))

# ── 3. Analyse multiple files ─────────────────────────────
print("\n[3] Analyse Files (POST /api/v1/analyze_focused)")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
py_files = []
for dirpath, dirs, files in os.walk(os.path.join(project_root, "backend")):
    dirs[:] = [d for d in dirs if d not in [ "__pycache__", ".venv", "venv"]]
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(dirpath, f).replace("\\", "/"))

test_file = py_files[0] if py_files else ""

for fp in py_files[:5]: # Test a subset to save time
    try:
        encoded = urllib.parse.quote(fp, safe="")
        code, body = post(f"/api/v1/analyze_focused?file_path={encoded}")
        data = json.loads(body)
        log(code == 200, f"Focused Analyse {os.path.basename(fp)}", f"nodes={len(data.get('nodes', []))} summary_len={len(data.get('summary', ''))}")
    except Exception as e:
        log(False, f"Focused Analyse {os.path.basename(fp)}", str(e))

# ── 4. Impact Analysis ────────────────────────────────────
print("\n[4] Impact Analysis")
if test_file:
    try:
        url = BASE + "/api/v1/impact_analysis"
        payload = json.dumps({
            "file_path": test_file,
            "change_description": "change array to list in the main loop"
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read().decode())
        log(resp.status == 200, "POST /api/v1/impact_analysis", f"affected={len(data.get('affected_node_ids', []))}")
    except Exception as e:
        log(False, "POST /api/v1/impact_analysis", str(e))

# ── 5. Explain Node ───────────────────────────────────────
print("\n[5] Explain Node")
if test_file:
    try:
        url = BASE + "/api/v1/explain_node"
        payload = json.dumps({
            "node_id": f"{test_file}::test",
            "label": "test",
            "file_path": test_file,
            "fragility": 50.0
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read().decode())
        log(resp.status == 200, "POST /api/v1/explain_node", f"explanation_len={len(data.get('explanation', ''))}")
    except Exception as e:
        log(False, "POST /api/v1/explain_node", str(e))

# ── 6. Graph Data ─────────────────────────────────────────
print("\n[6] Graph Data")
try:
    code, body = get("/api/v1/graph_data")
    data = json.loads(body)
    log(code == 200, "GET /api/v1/graph_data", f"nodes={len(data.get('nodes', []))}")
except Exception as e:
    log(False, "GET /api/v1/graph_data", str(e))

# ── 7. Frontend Static Files ──────────────────────────────
print("\n[7] Frontend Serving")
for path in ["/", "/style.css", "/app.js"]:
    try:
        code, body = get(path)
        log(code == 200, f"GET {path}", f"{len(body)} bytes")
    except Exception as e:
        log(False, f"GET {path}", str(e))

# ── 7. WebSocket ──────────────────────────────────────────
print("\n[7] WebSocket /ws")
try:
    import websockets
    import asyncio

    async def test_ws():
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as ws_conn:
            # Send a code change payload
            payload = {
                "file_path": test_file,
                "content": open(test_file, "r", encoding="utf-8").read() if test_file else "def foo(): pass",
            }
            await ws_conn.send(json.dumps(payload))
            resp = await asyncio.wait_for(ws_conn.recv(), timeout=30)
            data = json.loads(resp)
            return data

    ws_data = asyncio.run(test_ws())
    has_type = ws_data.get("type") == "fragility_update"
    has_nodes = isinstance(ws_data.get("nodes"), list)
    log(has_type and has_nodes, "WebSocket /ws", f"type={ws_data.get('type')}, nodes={len(ws_data.get('nodes', []))}, edges={len(ws_data.get('edges', []))}")
except ImportError:
    log(False, "WebSocket /ws", "websockets library not installed")
except Exception as e:
    log(False, "WebSocket /ws", str(e))

# ── Summary ───────────────────────────────────────────────
print(f"\n{'=' * 50}")
print(f"  RESULTS:  {PASS} passed  |  {FAIL} failed")
print(f"{'=' * 50}\n")
