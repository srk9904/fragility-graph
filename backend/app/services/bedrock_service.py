import boto3
import json
import logging
from .config import settings

logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self):
        self.model_id = settings.BEDROCK_MODEL_ID
        self.region = settings.AWS_REGION
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.AWS_ACCESS_KEY_ID != "dummy" else None,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.AWS_SECRET_ACCESS_KEY != "dummy" else None,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            self.client = None

    async def get_fragility_explanation(self, node_name: str, impact_count: int, dependencies: list):
        """
        Generates a concise 2-sentence explanation of why a node is fragile.
        """
        if not self.client:
            return "AI Explanation unavailable: Bedrock client not initialized."

        dependency_str = ", ".join(dependencies[:5])
        prompt = f"""
You are an AI assistant inside a developer tool. 

Explain in exactly 2 concise sentences why modifying '{node_name}' is risky.
Context: It has {impact_count} direct dependents and affects {dependency_str}.

Focus only on dependency impact and system failures. 
Do not list points. Do not give examples. Keep it short and direct.
"""
        
        try:
            # Format payload based on model family
            if "nova" in self.model_id.lower():
                payload = {
                    "inferenceConfig": {"maxTokens": 150, "temperature": 0.5},
                    "messages": [{"role": "user", "content": [{"text": prompt}]}]
                }
            else: # Claude 3 fallback
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}]
                }

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response.get('body').read())
            
            # Extract text based on model
            if "nova" in self.model_id.lower():
                return response_body['output']['message']['content'][0]['text'].strip()
            else:
                return response_body['content'][0]['text'].strip()

        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return f"AI Explanation unavailable: Error connecting to model."

bedrock_service = BedrockService()
