"""
Faithfulness metric evaluator.
Calculates the numerical index representing answer grounding correctness relative to retrieved context.
"""

import logging
from validation.citation_validator import CitationValidator
from validation.nli_validator import NLIValidator

logger = logging.getLogger(__name__)

class FaithfulnessMetric:
    def __init__(self, nli_validator: NLIValidator = None):
        self.validator = CitationValidator(nli_validator=nli_validator)

    def calculate_score(self, query: str, response: str, contexts: list[dict]) -> float:
        """
        Computes the ratio of logically entailed statements within the generated response.
        
        Args:
            query (str): Original search query.
            response (str): System generated response.
            contexts (list[dict]): Core context documents utilized.
            
        Returns:
            float: Score bounded between 0.0 (ungrounded) and 1.0 (fully grounded).
        """
        if not response:
            return 0.0
            
        # Refusal states are implicitly considered faithful (they state they don't know rather than lying)
        if "i cannot find supporting evidence" in response.lower() or "insufficient_evidence" in response:
            logger.debug("Refusal detected. Score evaluates to 1.0 (Faithful refusal).")
            return 1.0

        try:
            results = self.validator.validate_answer(response, contexts)
            score = results.get("faithfulness_score", 0.0)
            logger.debug(f"Calculated Faithfulness Score: {score:.4f} (Verified claims: {len(results.get('verified_claims', []))}/{results.get('total_claims', 0)})")
            return score
        except Exception as e:
            logger.error(f"Error calculating faithfulness score: {e}")
            return 0.0
