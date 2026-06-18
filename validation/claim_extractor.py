"""
Sentence-Level Claim Extractor.
Parses raw LLM text outputs into individual factual claims 
and extracts matching inline citation brackets.
"""

import re
import logging

logger = logging.getLogger(__name__)

class ClaimExtractor:
    def __init__(self):
        # Match keys e.g. [paper_001:methodology:4]
        self.citation_pattern = re.compile(
            r"\[([a-zA-Z0-9_\-\.]+):([a-zA-Z0-9_\-\.]+):([0-9]+)\]"
        )

    def extract_claims(self, text: str) -> list[dict]:
        """
        Splits text into claims and compiles corresponding citation anchors.
        
        Args:
            text (str): Raw LLM response string.
            
        Returns:
            list[dict]: Array of parsed claims containing 'clean_claim', 'raw_sentence', and 'citations'.
        """
        if not text:
            return []
            
        # Split on sentence boundaries (period/exclamation/question mark followed by space)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        parsed_claims = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Find all inline citation tags in the sentence
            citations = self.citation_pattern.findall(sentence)
            
            # Strip citation tags out to get the raw assertion text for NLI checks
            clean_sentence = self.citation_pattern.sub("", sentence).strip()
            # Clean double spaces or leading punctuation left after strip
            clean_sentence = re.sub(r"\s+", " ", clean_sentence)
            
            parsed_claims.append({
                "raw_sentence": sentence.strip(),
                "clean_claim": clean_sentence,
                "citations": citations  # list of tuples: (paper_id, section, page)
            })
            
        logger.debug(f"Extracted {len(parsed_claims)} claims from response text.")
        return parsed_claims
