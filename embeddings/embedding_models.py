"""
Embedding Model setup and environments.
Configures model names, dimension layers, and local cache directories.
"""

import os
import logging
from configs.config_manager import get_config

logger = logging.getLogger(__name__)

def setup_embedding_environment():
    """
    Configures HuggingFace caching parameters based on SystemConfig.
    """
    config = get_config()
    
    # We read HF_HOME cache override from environment
    hf_cache = os.getenv("HF_HOME", "./data/embeddings/hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    
    # Disable tokenizers parallelism warning logs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logger.info(f"HuggingFace model environment cache set to: {hf_cache}")
    
    return {
        "model_name": config.embedding.model_name,
        "dimension": config.embedding.dimension,
        "query_prefix": config.embedding.query_prefix
    }
