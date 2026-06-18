"""
Context Deduplication module.
Removes redundant text segments from retrieved context arrays using embedding cosine similarity 
or lexical token Jaccard similarity as a fallback.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes Jaccard similarity between two strings using token sets."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a or set_b else 0.0

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes Cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def deduplicate_chunks(chunks: List[Dict], threshold: float = 0.90) -> List[Dict]:
    """Filters out semantically duplicate chunks based on similarity threshold."""
    unique_chunks = []
    
    for candidate in chunks:
        is_duplicate = False
        cand_embed = candidate.get("embedding")
        cand_text = candidate.get("content", "")
        
        for unique_chunk in unique_chunks:
            uniq_embed = unique_chunk.get("embedding")
            uniq_text = unique_chunk.get("content", "")
            
            if cand_embed is not None and uniq_embed is not None:
                try:
                    similarity = cosine_similarity(cand_embed, uniq_embed)
                except Exception as e:
                    logger.warning(f"Cosine similarity error: {e}. Falling back to Jaccard.")
                    similarity = jaccard_similarity(cand_text, uniq_text)
            else:
                similarity = jaccard_similarity(cand_text, uniq_text)
                
            if similarity > threshold:
                is_duplicate = True
                logger.debug(f"Filtered duplicate chunk '{candidate.get('chunk_id')}' (similarity: {similarity:.4f}).")
                break
                
        if not is_duplicate:
            unique_chunks.append(candidate)
            
    return unique_chunks
