import os
import json
import asyncio
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.services.ml_service import ml_service

async def run_suite():
    test_dir = os.path.join(os.path.dirname(__file__), "test_cases")
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".json")]
    
    print(f"--- FragilityGraph Automation Suite: Running {len(test_files)} Tests ---")
    
    passed = 0
    for test_file in test_files:
        with open(os.path.join(test_dir, test_file), "r") as f:
            t = json.load(f)
        
        print(f"\n[TEST] {t['name']}: {t['description']}")
        
        results = ml_service.predict(t['input'])
        
        # Simple validation
        success = True
        for res in results:
            score = res['score']
            if not (t['expected']['min_score'] <= score <= t['expected']['max_score']):
                print(f"  FAILED: Node {res['name']} score {score} outside range [{t['expected']['min_score']}, {t['expected']['max_score']}]")
                success = False
            else:
                print(f"  PASSED: Node {res['name']} score {score} hits targets.")

        if success: passed += 1

    print(f"\n--- Suite Finished: {passed}/{len(test_files)} Passed ---")

if __name__ == "__main__":
    asyncio.run(run_suite())
