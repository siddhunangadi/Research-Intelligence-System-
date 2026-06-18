"""
Refusal and safety intercept coordinator.
Interprets retrieval outputs and citation validations to trigger standard 
grounding refutation strings in place of conversational hallucinations.
"""

import logging

logger = logging.getLogger(__name__)

STANDARD_REFUSAL = "I cannot find supporting evidence within the provided corpus to answer this query."

class RefusalHandler:
    def __init__(self, fallback_message: str = STANDARD_REFUSAL):
        self.fallback_message = fallback_message

    def is_refusal_trigger(self, raw_answer: str) -> bool:
        """
        Detects if the LLM output indicates it cannot answer the query.
        """
        refusal_markers = [
            "insufficient_evidence",
            "insufficient evidence",
            "i cannot answer",
            "i do not have access",
            "not found in the provided sources",
            "no matching context",
            "provided documents do not specify"
        ]
        answer_lower = raw_answer.lower().strip()
        
        # Check standard indicators
        for marker in refusal_markers:
            if marker in answer_lower:
                return True
                
        # If the LLM returned a blank or extremely short response, consider it a refusal state
        if len(answer_lower) < 5:
            return True
            
        return False

    def process_response(
        self, 
        raw_answer: str, 
        validation_results: dict, 
        retrieved_chunks: list[dict]
    ) -> str:
        """
        Applies zero-trust refusal conditions over RAG generation results.
        
        Refusal conditions:
        1. Retrieval returned no viable segments.
        2. LLM explicitly flagged inability to answer.
        3. More than 50% of the generated assertions failed NLI validation.
        
        Args:
            raw_answer (str): Text output from LLM.
            validation_results (dict): Diagnostic dictionary returned by CitationValidator.
            retrieved_chunks (list[dict]): Selected candidate contexts.
            
        Returns:
            str: Refined grounded answer or standard refusal response.
        """
        # Condition 1: No source documents retrieved
        if not retrieved_chunks:
            logger.warning("Refusal Triggered: Chunks array is empty.")
            return self.fallback_message

        # Condition 2: LLM emitted text matches explicit refusal trigger phrases
        if self.is_refusal_trigger(raw_answer):
            logger.info("Refusal Triggered: LLM emitted standard rejection token.")
            return self.fallback_message

        # Condition 3: Grounding validation fails catastrophically (e.g. less than 50% faithfulness)
        faithfulness = validation_results.get("faithfulness_score", 0.0)
        if faithfulness < 0.50:
            logger.warning(f"Refusal Triggered: Faithfulness score is too low ({faithfulness:.2f} < 0.50). Rejecting answer.")
            return self.fallback_message

        # If clean, return the raw answer. (Can optionally strip remaining citation tags if clean output is preferred,
        # but for academic integrity we keep inline source markers).
        return raw_answer
