"""
Vector Embedder module.
Loads BGE-Large-v1.5 model locally to generate 1024-dimensional embeddings, 
handling prefix injection and fallback mocks for environment portability.
"""

import logging
import hashlib
import numpy as np
from embeddings.embedding_models import setup_embedding_environment

logger = logging.getLogger(__name__)

# Fallback checker
HAS_TORCH = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TORCH = True
except ImportError:
    logger.warning("transformers/torch not installed. Embedder running in Mock mode.")

class ScientificEmbedder:
    def __init__(self):
        env_params = setup_embedding_environment()
        self.model_name = env_params["model_name"]
        self.dimension = env_params["dimension"]
        self.query_prefix = env_params["query_prefix"]
        
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        
        if HAS_TORCH:
            try:
                self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                logger.info(f"Loading local embedding weights '{self.model_name}' on device: {self.device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error(f"Error loading model weights: {e}. Falling back to mock embedder.")
                self.model = None

    def _generate_mock_vector(self, text: str) -> list[float]:
        """
        Generates a deterministic, normalized mock vector of size 1024.
        """
        # Create seed hash from text
        hash_val = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed numpy generator for determinism
        seed = int.from_bytes(hash_val[:4], byteorder="big")
        rng = np.random.default_rng(seed)
        
        # Generate random values and normalize
        vec = rng.standard_normal(self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generates vector representations for a batch of documents.
        """
        if not texts:
            return []
            
        if not self.model or not HAS_TORCH:
            return [self._generate_mock_vector(t) for t in texts]

        try:
            # Tokenize inputs
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Compute mean pooling (standard for BGE models)
                token_embeddings = model_output[0]
                input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                # L2 normalization for cosine similarity search
                normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            return normalized_embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}. Falling back to mock vectors.")
            return [self._generate_mock_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """
        Generates vector representation for a query, adding the BGE asymmetric search prefix.
        """
        prefixed_query = f"{self.query_prefix}{query}"
        results = self.embed_documents([prefixed_query])
        return results[0] if results else [0.0] * self.dimension
