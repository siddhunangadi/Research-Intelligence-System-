"""
Retrieval Metrics Calculator.
Provides mathematical implementations for Precision@K, Recall@K, 
Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG).
"""

import math
import logging

logger = logging.getLogger(__name__)

def calculate_precision_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> float:
    """
    Precision@K = (Relevant Chunks retrieved inside Top K) / K
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0
        
    sliced_retrieved = retrieved_ids[:k]
    relevant_retrieved = set(sliced_retrieved).intersection(set(ground_truth_ids))
    return len(relevant_retrieved) / k

def calculate_recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> float:
    """
    Recall@K = (Relevant Chunks retrieved inside Top K) / (Total Relevant Chunks)
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0
        
    sliced_retrieved = retrieved_ids[:k]
    relevant_retrieved = set(sliced_retrieved).intersection(set(ground_truth_ids))
    return len(relevant_retrieved) / len(ground_truth_ids)

def calculate_mrr(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank (MRR) = 1 / rank of first relevant chunk.
    Returns 0.0 if no relevant chunk is found.
    """
    if not retrieved_ids or not ground_truth_ids:
        return 0.0
        
    for index, rid in enumerate(retrieved_ids, start=1):
        if rid in ground_truth_ids:
            return 1.0 / index
            
    return 0.0

def calculate_ndcg(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) = DCG@K / IDCG@K
    Where DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0
        
    sliced_retrieved = retrieved_ids[:k]
    
    # Calculate DCG@K (relevance binary check: 1 if relevant else 0)
    dcg = 0.0
    for idx, rid in enumerate(sliced_retrieved, start=1):
        rel = 1.0 if rid in ground_truth_ids else 0.0
        dcg += rel / math.log2(idx + 1)
        
    # Calculate Ideal DCG@K (all ground truths placed at the top)
    idcg = 0.0
    # Ideal score places all matched targets in the top ranks (up to k elements)
    ideal_match_count = min(len(ground_truth_ids), k)
    for idx in range(1, ideal_match_count + 1):
        idcg += 1.0 / math.log2(idx + 1)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
