import boto3
import logging
import os
from .config import settings

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.bucket_name = settings.AWS_S3_BUCKET_NAME
        try:
            self.client = boto3.client(
                's3',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.AWS_ACCESS_KEY_ID != "dummy" else None,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.AWS_SECRET_ACCESS_KEY != "dummy" else None,
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.client = None

    def list_models(self):
        """List available model weights in S3."""
        if not self.client: return []
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix="models/")
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"S3 list_models failed: {e}")
            return []

    def download_model(self, model_key, local_path):
        """Downloads model weights for GNN inference."""
        if not self.client: return False
        try:
            logger.info(f"Downloading model {model_key} from S3...")
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.download_file(self.bucket_name, model_key, local_path)
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

s3_service = S3Service()
