"""
Context Precision Metric.
Calculates the rank density and relevance positioning of context chunks 
passed in prompt windows.
"""

import logging
from evaluation.metrics.retrieval_metrics import calculate_precision_at_k

logger = logging.getLogger(__name__)

class ContextPrecisionMetric:
    def calculate_context_precision(self, retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
        """
        Computes context precision: evaluates if target chunks are placed high in the rank listing.
        
        Formula:
            ContextPrecision = sum_{k=1}^K (Precision@k * rel(k)) / (Total Relevant Chunks in Top K)
        """
        if not retrieved_ids or not ground_truth_ids:
            return 0.0
            
        relevant_retrieved = set(retrieved_ids).intersection(set(ground_truth_ids))
        if not relevant_retrieved:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0
        
        for k, rid in enumerate(retrieved_ids, start=1):
            if rid in ground_truth_ids:
                relevant_count += 1
                precision_at_k = calculate_precision_at_k(retrieved_ids, ground_truth_ids, k)
                precision_sum += precision_at_k
                
        if relevant_count == 0:
            return 0.0
            
        return precision_sum / relevant_count
