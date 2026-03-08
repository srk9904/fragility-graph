import os
import sys

# Minimal .env parser so we don't need python-dotenv for the test script
def load_env():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
    if not os.path.exists(env_path):
        print(f"[WARN] No .env file found at {env_path}")
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key and value:
                os.environ[key.strip()] = value.strip()

# Try to load required libraries safely
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    import redis
except ImportError:
    redis = None

def test_aws():
    print("Testing AWS Connection...")
    if not boto3:
        print("  [SKIP] 'boto3' library is not installed. Run: pip install boto3")
        return
    
    region = os.getenv("AWS_REGION")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    
    try:
        if access_key and secret_key:
            kwargs = {
                'region_name': region,
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key
            }
            if session_token:
                kwargs['aws_session_token'] = session_token
                
            sts = boto3.client('sts', **kwargs)
        else:
            # Fallback to local profile config
            sts = boto3.client('sts')
            
        response = sts.get_caller_identity()
        print(f"  [SUCCESS] Connected to AWS. Account ID: {response.get('Account')}")
    except ClientError as e:
        print(f"  [FAILED] AWS Connection error: {e}")
    except Exception as e:
        print(f"  [FAILED] AWS error: {e}")

def test_neo4j():
    print("\nTesting Neo4j Connection...")
    if not GraphDatabase:
        print("  [SKIP] 'neo4j' library is not installed. Run: pip install neo4j")
        return
        
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri:
        print("  [SKIP] NEO4J_URI not set in .env")
        return
        
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"  [SUCCESS] Successfully connected to Neo4j as user: {user}.")
        driver.close()
    except Exception as e:
        print(f"  [FAILED] Neo4j connection failed as user {user}: {e}")

def test_redis():
    print("\nTesting Redis Connection...")
    if not redis:
        print("  [SKIP] 'redis' library is not installed. Run: pip install redis")
        return
        
    url = os.getenv("REDIS_URL")
    if not url:
        print("  [SKIP] REDIS_URL not set in .env")
        return
        
    try:
        r = redis.from_url(url)
        if r.ping():
            print("  [SUCCESS] Successfully connected to Redis.")
        else:
            print("  [FAILED] Redis ping failed without exception.")
    except Exception as e:
        print(f"  [FAILED] Redis connection failed: {e}")

if __name__ == "__main__":
    print("====================================")
    print("     Connection Tests Runner        ")
    print("====================================\n")
    
    load_env()
    
    test_aws()
    test_neo4j()
    test_redis()
    
    print("\n====================================")
    print("             Done                   ")
    print("====================================")
