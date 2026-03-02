import sys
import os
import asyncio
import json

# Add the project root to sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_env():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if not os.path.exists(root_env):
        print(f"DEBUG: .env not found at {root_env}")
        return
    with open(root_env, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

load_env()
from backend.app.services.bedrock_service import bedrock_service

async def test_bedrock_service():
    print("--- Testing Optimized Bedrock Service ---")
    
    node_name = "AuthModule.verify_token"
    impact_count = 12
    dependencies = ["Gateway", "UserDB", "SessionStore"]
    
    print(f"Node: {node_name}")
    print(f"Dependents: {impact_count}")
    print(f"Deps: {dependencies}")
    print("\nRequesting explanation...")
    
    explanation = await bedrock_service.get_fragility_explanation(node_name, impact_count, dependencies)
    
    border = "=" * 60
    print(f"\n{border}")
    print("AI OUTPUT (Product View):")
    print(f"{border}")
    print(explanation)
    print(f"{border}\n")

if __name__ == "__main__":
    if os.getenv('BEDROCK_MODEL_ID') is None:
        print("ERROR: BEDROCK_MODEL_ID not set in environment.")
        sys.exit(1)
        
    asyncio.run(test_bedrock_service())
