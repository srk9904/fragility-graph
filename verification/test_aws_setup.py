import boto3
import sys
import os
from botocore.exceptions import ClientError, NoCredentialsError

def load_env(file_path=".env"):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

load_env()

def test_s3(bucket_name):
    print(f"\n[1/3] Testing S3 access for bucket: {bucket_name}")
    s3 = boto3.client('s3')
    try:
        s3.head_bucket(Bucket=bucket_name)
        print("SUCCESS: S3 bucket exists and is accessible.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"ERROR: Bucket '{bucket_name}' not found.")
        elif error_code == '403':
            print(f"ERROR: Access denied to bucket '{bucket_name}'. Check IAM permissions.")
        else:
            print(f"ERROR: S3 Error: {e}")
    except NoCredentialsError:
        print("ERROR: No AWS credentials found. Run 'aws configure'.")

def test_bedrock_direct(model_id):
    print(f"\n[2b/3] Testing DIRECT Bedrock Invocation: {model_id}")
    runtime = boto3.client('bedrock-runtime')
    try:
        import json
        if 'anthropic' in model_id.lower():
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        elif 'nova' in model_id.lower():
            payload = {
                "inferenceConfig": {"maxTokens": 10},
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}]
            }
        else: # Titan
            payload = {
                "inputText": "Hello",
                "textGenerationConfig": {"maxTokenCount": 10}
            }

        runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(payload)
        )
        print(f"SUCCESS: Direct invocation of '{model_id}' worked!")
    except ClientError as e:
        print(f"ERROR: Direct Invocation failed: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")

def test_bedrock(model_id):
    print(f"\n[2a/3] Testing Bedrock model access (List): {model_id}")
    bedrock_admin = boto3.client('bedrock')
    try:
        models = bedrock_admin.list_foundation_models()
        found = any(m['modelId'] == model_id for m in models.get('modelSummaryList', []))
        
        if found:
            print(f"SUCCESS: Model '{model_id}' is available in your account/region.")
        else:
            print(f"WARNING: Model '{model_id}' not found in list_foundation_models.")
            print(f"Available 'anthropic' models in {bedrock_admin.meta.region_name}:")
            for m in models.get('modelSummaryList', []):
                if 'anthropic' in m['modelId'].lower():
                    print(f" - {m['modelId']}")
            
            # If not found in list, try direct invocation anyway
            test_bedrock_direct(model_id)
            
    except ClientError as e:
        print(f"ERROR: Bedrock List Error: {e}")
        # Try direct invocation even if list fails
        test_bedrock_direct(model_id)
    except NoCredentialsError:
        print("ERROR: No AWS credentials found.")

def test_iam():
    print("\n[3/3] Testing IAM Identity")
    sts = boto3.client('sts')
    try:
        response = sts.get_caller_identity()
        print(f"SUCCESS: Authenticated as IAM User: {response['Arn']}")
    except ClientError as e:
        print(f"ERROR: IAM/STS Error: {e}")
    except NoCredentialsError:
        print("ERROR: No AWS credentials found.")

if __name__ == "__main__":
    load_env()
    # Load from environment or use placeholders
    bucket = os.getenv('AWS_S3_BUCKET_NAME', 'your-bucket-name')
    
    # Try Anthropic first, then Amazon Nova as a backup
    models_to_test = [
        os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0'),
        'amazon.nova-lite-v1:0',
        'amazon.nova-micro-v1:0'
    ]
    
    print("--- AWS Setup Verification ---")
    test_iam()
    test_s3(bucket)
    for model in models_to_test:
        test_bedrock(model)
    print("\n--- Verification Complete ---")
