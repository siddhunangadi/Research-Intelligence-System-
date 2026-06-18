"""
Cross-Encoder Reranker using ms-marco-MiniLM-L-6-v2.
Refines candidate chunk ordering by evaluating query-chunk relevance dynamically.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers or torch not installed. Running Reranker in Mock mode.")

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", threshold: float = 0.35):
        self.model_name = model_name
        self.threshold = threshold
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        
        if HAS_TRANSFORMERS:
            try:
                self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                logger.info(f"Loading Cross-Encoder Reranker '{self.model_name}' on device: {self.device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error(f"Error loading Cross-Encoder model: {e}. Falling back to mock reranker.")
                self.model = None

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 8) -> List[Dict]:
        """Reranks retrieved candidate chunks based on Cross-Encoder scoring."""
        if not chunks:
            return []
            
        if not self.model or not HAS_TRANSFORMERS:
            logger.warning("Reranker running in mock mode due to missing dependencies/model initialization.")
            reranked = []
            for i, chunk in enumerate(chunks):
                chunk_copy = dict(chunk)
                # Assign dummy score degrading from 0.9 down to mock threshold
                chunk_copy["rerank_score"] = max(1.0 - (i * 0.05), 0.1)
                if chunk_copy["rerank_score"] >= self.threshold:
                    reranked.append(chunk_copy)
            return reranked[:top_k]

        pairs = [[query, chunk.get("content", "")] for chunk in chunks]
        
        try:
            with torch.no_grad():
                features = self.tokenizer(
                    pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**features)
                scores = outputs.logits.squeeze(-1)
                
                probs = torch.sigmoid(outputs.logits).squeeze(-1).unsqueeze(0) if len(chunks) == 1 else torch.sigmoid(scores)
                scores_list = probs.cpu().numpy().tolist()
                if not isinstance(scores_list, list):
                    scores_list = [scores_list]
        except Exception as e:
            logger.error(f"Inference error during reranking: {e}. Preserving original ranks.")
            return chunks[:top_k]

        scored_chunks = [{**chunks[idx], "rerank_score": float(score)} for idx, score in enumerate(scores_list)]
        sorted_chunks = sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True)
        filtered_chunks = [c for c in sorted_chunks if c["rerank_score"] >= self.threshold]
        
        logger.debug(f"Reranking filtered {len(chunks)} chunks down to {len(filtered_chunks)} above threshold {self.threshold}.")
        return filtered_chunks[:top_k]
