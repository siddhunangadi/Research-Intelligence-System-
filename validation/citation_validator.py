"""
Citation and claim verification engine.
Parses generated answers into sentences, extracts inline citations, 
retrieves the premise text from matched sources, and runs validation checking.
"""

import re
import logging
from validation.nli_validator import NLIValidator
from validation.claim_extractor import ClaimExtractor

logger = logging.getLogger(__name__)

class CitationValidator:
    def __init__(self, nli_validator: NLIValidator = None):
        self.nli_validator = nli_validator or NLIValidator()
        self.extractor = ClaimExtractor()

    def validate_answer(self, answer: str, retrieved_chunks: list[dict]) -> dict:
        """
        Validates an answer by comparing each sentence claim to its cited chunk context.
        
        Args:
            answer (str): The raw text response from the LLM.
            retrieved_chunks (list[dict]): Chunks that were passed in the prompt context.
            
        Returns:
            dict: Evaluation results showing verified/unverified statements and aggregate scores.
        """
        claims = self.extractor.extract_claims(answer)
        
        # Map retrieved chunks by citation keys for fast lookups
        chunk_map = {}
        for chunk in retrieved_chunks:
            paper_id = chunk.get("paper_id", "")
            section = chunk.get("section", "")
            page = str(chunk.get("page_number", ""))
            key = f"{paper_id}:{section}:{page}"
            chunk_map[key] = chunk.get("content", "")

        verified_claims = []
        unverified_claims = []
        contradicted_claims = []
        
        total_claims_with_citations = 0
        total_passed = 0

        for claim_data in claims:
            citations = claim_data["citations"]
            clean_claim = claim_data["clean_claim"]
            
            # If no citations were provided, flag it as unverified
            if not citations:
                unverified_claims.append({
                    "claim": claim_data,
                    "reason": "No citation anchors found for this assertion."
                })
                continue
                
            total_claims_with_citations += 1
            claim_passed_any_citation = False
            reasons = []
            
            # A claim can cite multiple chunks. It passes validation if *at least one* cited chunk
            # logically entails it.
            for paper_id, section, page in citations:
                citation_key = f"{paper_id}:{section}:{page}"
                premise = chunk_map.get(citation_key)
                
                if not premise:
                    # In cases where the citation key does not match the chunks retrieved, try to match by paper_id only
                    similar_premises = [content for key, content in chunk_map.items() if key.startswith(f"{paper_id}:")]
                    if similar_premises:
                        premise = "\n".join(similar_premises)
                    else:
                        reasons.append(f"Citation key '{citation_key}' did not match retrieved context.")
                        continue
                
                # Check logic alignment via NLI model
                nli_result = self.nli_validator.check_entailment(premise, clean_claim)
                
                if nli_result["passed"]:
                    claim_passed_any_citation = True
                    verified_claims.append({
                        "claim": claim_data,
                        "verified_by": citation_key,
                        "score": nli_result["entailment_probability"]
                    })
                    break
                else:
                    reasons.append(f"Verification against '{citation_key}' failed (NLI label: {nli_result['label']}, Score: {nli_result['entailment_probability']:.2f}).")
                    if nli_result["label"] == "contradiction":
                        contradicted_claims.append({
                            "claim": claim_data,
                            "contradicted_by": citation_key,
                            "score": nli_result["entailment_probability"]
                        })

            if claim_passed_any_citation:
                total_passed += 1
            else:
                unverified_claims.append({
                    "claim": claim_data,
                    "reason": "; ".join(reasons)
                })

        # Calculate grounding ratio based on citation targets
        grounding_ratio = total_passed / max(total_claims_with_citations, 1) if total_claims_with_citations > 0 else 0.0
        
        # Absolute faithfulness (including un-cited statements in the denominator)
        faithfulness_score = total_passed / max(len(claims), 1)
        
        passed_validation = faithfulness_score >= 0.85
        
        return {
            "is_faithful": passed_validation,
            "faithfulness_score": faithfulness_score,
            "grounding_ratio": grounding_ratio,
            "total_claims": len(claims),
            "claims_with_citations": total_claims_with_citations,
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims,
            "contradicted_claims": contradicted_claims
        }
