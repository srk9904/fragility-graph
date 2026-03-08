import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AWS
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_SESSION_TOKEN: str = ""
    AWS_S3_BUCKET_NAME: str = ""
    BEDROCK_MODEL_ID: str = "amazon.nova-lite-v1:0"

    # Neo4j
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""

    # Redis
    REDIS_URL: str = ""

    # App
    BACKEND_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # Project root to scan for file tree
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    model_config = SettingsConfigDict(
        env_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
