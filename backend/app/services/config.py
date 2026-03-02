from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # AWS
    AWS_ACCESS_KEY_ID: str = "dummy"
    AWS_SECRET_ACCESS_KEY: str = "dummy"
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET_NAME: str = "dummy-bucket"
    BEDROCK_MODEL_ID: str = "amazon.nova-lite-v1:0"
    
    # Neo4j
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    
    # General
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
