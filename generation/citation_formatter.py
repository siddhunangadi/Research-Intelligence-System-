"""
Citation Formatting Utilities.
Corrects period coordinates relative to citation brackets 
to prevent boundary-split failures in verification.
"""

import re
import logging

logger = logging.getLogger(__name__)

class CitationFormatter:
    def __init__(self):
        # Matches patterns like ". [arxiv_1706_03762:methodology:4]" or ".  [paper_abc:results:1]"
        self.misplaced_period_pattern = re.compile(
            r"\.\s*(\[[a-zA-Z0-9_\-\.]+:[a-zA-Z0-9_\-\.]+:[0-9]+\])"
        )

    def clean_citation_periods(self, text: str) -> str:
        """
        Reorders period placement from before to after the bracket.
        
        Example input:
            "This model is trained with AdamW. [arxiv_1706_03762:methodology:4]"
        Example output:
            "This model is trained with AdamW [arxiv_1706_03762:methodology:4]."
        """
        # We replace ". [bracket]" with " [bracket]."
        cleaned_text = self.misplaced_period_pattern.sub(r" \1.", text)
        
        # Clean double spacing anomalies
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        
        if cleaned_text != text:
            logger.debug("Corrected misplaced periods before citation tags.")
            
        return cleaned_text.strip()
        
    def generate_citation_tag(self, paper_id: str, section: str, page: int) -> str:
        """
        Synthesizes standard string citation token.
        """
        return f"[{paper_id}:{section}:{page}]"
