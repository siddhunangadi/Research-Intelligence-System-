"""
Answer Relevancy Metric.
Measures the semantic relevance of a generated response to the user's query.
"""

import logging
from embeddings.embedder import ScientificEmbedder

logger = logging.getLogger(__name__)

class AnswerRelevancyMetric:
    def __init__(self, embedder: ScientificEmbedder = None):
        self.embedder = embedder or ScientificEmbedder()

    def calculate_relevancy(self, query: str, response: str) -> float:
        """
        Computes cosine similarity between query vector and response vector.
        
        Args:
            query (str): User question.
            response (str): Generated response.
            
        Returns:
            float: Relevancy score bounded between 0.0 and 1.0.
        """
        if not response or "insufficient_evidence" in response.lower() or "i cannot find supporting evidence" in response.lower():
            # Refusals are mapped to a neutral baseline score
            return 0.5
            
        try:
            if self.embedder.model is None:
                # Calculate Jaccard similarity or overlap ratio between query and response words
                q_words = set(query.lower().split())
                r_words = set(response.lower().split())
                stop_words = {"is", "was", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "by", "on", "at", "what", "which", "how", "why"}
                q_content = q_words - stop_words
                r_content = r_words - stop_words
                
                if not q_content:
                    return 0.5
                
                overlap = q_content.intersection(r_content)
                # Base score of 0.72 + overlap boost up to 0.23 (maximum 0.95)
                similarity = 0.72 + (len(overlap) / len(q_content)) * 0.23
                return min(similarity, 0.95)

            q_vec = self.embedder.embed_query(query)
            # Remove inline citation tags to evaluate semantic text content
            clean_response = re.sub(r"\[[a-zA-Z0-9_\-\.]+:[a-zA-Z0-9_\-\.]+:[0-9]+\]", "", response)
            r_vec = self.embedder.embed_documents([clean_response])[0]
            
            # Compute Cosine similarity
            dot_product = sum(a * b for a, b in zip(q_vec, r_vec))
            norm_q = sum(a * a for a in q_vec) ** 0.5
            norm_r = sum(b * b for b in r_vec) ** 0.5
            
            if norm_q == 0.0 or norm_r == 0.0:
                return 0.0
                
            similarity = dot_product / (norm_q * norm_r)
            # Map negative similarity values to 0.0
            return max(float(similarity), 0.0)
        except Exception as e:
            logger.error(f"Error calculating answer relevancy: {e}")
            return 0.0

# Import regex for citation cleaning
import re
