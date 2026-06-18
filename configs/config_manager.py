"""
Configuration management engine.
Loads environment variables and YAML parameters into strongly typed Pydantic models.
"""

import os
import yaml
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ChunkingConfig(BaseModel):
    target_size: int = Field(default=700, ge=100)
    overlap: int = Field(default=100, ge=0)

class EmbeddingConfig(BaseModel):
    model_name: str = Field(default="BAAI/bge-large-en-v1.5")
    dimension: int = Field(default=1024, ge=1)
    query_prefix: str = Field(default="")

class RetrievalConfig(BaseModel):
    dense_top_k: int = Field(default=20, ge=1)
    sparse_top_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=8, ge=1)
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

class ContextConfig(BaseModel):
    max_tokens: int = Field(default=3000, ge=500)
    deduplication_threshold: float = Field(default=0.90, ge=0.0, le=1.0)

class ValidationConfig(BaseModel):
    nli_model: str = Field(default="cross-encoder/nli-deberta-v3-base")
    entailment_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

class DatabaseConfig(BaseModel):
    postgres_url: str
    chroma_path: str

class SystemConfig(BaseModel):
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    context: ContextConfig
    validation: ValidationConfig
    database: DatabaseConfig
    
    # API credentials
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_tracing: bool = False
    langchain_project: str = "research-intelligence-system"

class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, yaml_path: str = "configs/retrieval.yaml"):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.yaml_path = yaml_path
        self.config = self._load_config()
        self._initialized = True

    def _load_config(self) -> SystemConfig:
        """
        Parses YAML and merges with OS environment variables.
        """
        # Load local .env file manually to avoid dependency on python-dotenv
        env_path = ".env"
        if os.path.exists(env_path):
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            # Only set if not already set (retains test overrides)
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")


        # Load environment variables (usually seeded from .env)
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        mistral_key = os.getenv("MISTRAL_API_KEY")
        langsmith_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        langsmith_project = os.getenv("LANGCHAIN_PROJECT", "research-intelligence-system")

        
        pg_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/research_rag")
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")

        # Load YAML configuration
        yaml_data = {}
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, "r") as f:
                    yaml_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded YAML configuration from {self.yaml_path}")
            except Exception as e:
                logger.error(f"Error reading configuration YAML: {e}. Falling back to default settings.")
        else:
            logger.warning(f"Configuration file {self.yaml_path} not found. Using runtime defaults.")

        # Structure DatabaseConfig
        db_config = DatabaseConfig(
            postgres_url=pg_url,
            chroma_path=chroma_path
        )

        # Build combined configurations
        chunking = ChunkingConfig(**yaml_data.get("chunking", {}))
        embedding = EmbeddingConfig(**yaml_data.get("embedding", {}))
        retrieval = RetrievalConfig(**yaml_data.get("retrieval", {}))
        context = ContextConfig(**yaml_data.get("context", {}))
        validation = ValidationConfig(**yaml_data.get("validation", {}))

        return SystemConfig(
            chunking=chunking,
            embedding=embedding,
            retrieval=retrieval,
            context=context,
            validation=validation,
            database=db_config,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            mistral_api_key=mistral_key,
            langchain_api_key=langsmith_key,
            langchain_tracing=langsmith_tracing,
            langchain_project=langsmith_project
        )


# Global configuration instance
_manager = ConfigManager()
get_config = lambda: _manager.config
