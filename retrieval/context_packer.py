"""
Context Packer & Reordering module.
Reorders retrieved chunks to place the most relevant sources at the high-attention margins
of the LLM prompt context window (Beginning and End) and filters to stay within token budgets.
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Fallback checking for dependencies
HAS_TIKTOKEN = False
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    logger.warning("tiktoken not installed. Falling back to char-count-based token estimation.")

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Counts tokens for a given string."""
    if HAS_TIKTOKEN:
        try:
            return len(tiktoken.encoding_for_model(model).encode(text))
        except Exception:
            try:
                return len(tiktoken.get_encoding("cl100k_base").encode(text))
            except Exception:
                pass
    # Simple character estimation baseline (4 chars/token)
    return len(text) // 4

def lost_in_the_middle_reorder(chunks: List[Dict]) -> List[Dict]:
    """
    Reorders elements such that high-rank chunks are distributed on the boundaries 
    (beginning and end) and lower-rank chunks are located in the middle.
    """
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", x.get("rrf_score", 0)), reverse=True)
    reordered = [None] * len(sorted_chunks)
    
    left, right = 0, len(sorted_chunks) - 1
    for idx, item in enumerate(sorted_chunks):
        if idx % 2 == 0:
            reordered[left] = item
            left += 1
        else:
            reordered[right] = item
            right -= 1
            
    return reordered

def pack_context(
    chunks: List[Dict], 
    max_tokens: int = 3000, 
    model: str = "gpt-4o"
) -> Tuple[str, List[Dict]]:
    """Reorders chunks and packs them within a token budget limit."""
    reordered_chunks = lost_in_the_middle_reorder(chunks)
    packed_chunks = []
    current_tokens = 0
    formatted_blocks = []
    
    for chunk in reordered_chunks:
        paper_id = chunk.get("paper_id", "unknown")
        section = chunk.get("section", "unknown")
        page = chunk.get("page_number", "unknown")
        content = chunk.get("content", "").strip()
        
        block = f"\n--- SOURCE: [{paper_id}:{section}:{page}] ---\n{content}\n-------------------------------------\n"
        block_tokens = count_tokens(block, model)
        
        if current_tokens + block_tokens <= max_tokens:
            formatted_blocks.append(block)
            packed_chunks.append(chunk)
            current_tokens += block_tokens
        else:
            logger.debug(f"Token limit reached ({current_tokens}/{max_tokens} tokens). Skipping remaining chunks.")
            break
            
    return "".join(formatted_blocks), packed_chunks
