"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.
Unifies keyword-based search ranking (BM25) and dense semantic vector rankings.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def compute_rrf(
    sparse_results: List[Dict], 
    dense_results: List[Dict], 
    k: int = 60,
    top_n: int = 25
) -> List[Dict]:
    """Executes Reciprocal Rank Fusion (RRF) across sparse and dense candidates."""
    rrf_scores = {}
    
    # Accumulate reciprocal ranks for both sparse and dense outputs
    for rank_type, results in [("rank_sparse", sparse_results), ("rank_dense", dense_results)]:
        for rank, item in enumerate(results, start=1):
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "chunk": item,
                    "score": 0.0,
                    "rank_sparse": None,
                    "rank_dense": None
                }
            rrf_scores[chunk_id][rank_type] = rank
            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank)
        
    # Sort candidate chunks by fusion score descending
    sorted_candidates = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Map back to final unified candidates dictionary
    fused_results = []
    for item in sorted_candidates[:top_n]:
        chunk_data = {
            **item["chunk"],
            "rrf_score": item["score"],
            "rank_sparse": item["rank_sparse"],
            "rank_dense": item["rank_dense"]
        }
        fused_results.append(chunk_data)
        
    logger.debug(f"Fused {len(sparse_results)} sparse and {len(dense_results)} dense candidates into top {len(fused_results)} results.")
    return fused_results
