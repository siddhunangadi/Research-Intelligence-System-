"""
Full Retrieval, Generation, and Verification Pipeline workflow.
Integrates hybrid retrieval simulation, RRF fusion, Cross-Encoder reranking, 
context reordering, grounded answer checks, NLI citation verification, and refusal handling.
"""

import logging
from retrieval.rrf_fusion import compute_rrf
from retrieval.reranker import CrossEncoderReranker
from retrieval.deduplicator import deduplicate_chunks
from retrieval.context_packer import pack_context
from validation.nli_validator import NLIValidator
from validation.citation_validator import CitationValidator
from validation.refusal_handler import RefusalHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class ResearchRAGPipeline:
    def __init__(self):
        logger.info("Initializing Research RAG Pipeline components...")
        self.reranker = CrossEncoderReranker()
        self.nli_validator = NLIValidator()
        self.citation_validator = CitationValidator(nli_validator=self.nli_validator)
        self.refusal_handler = RefusalHandler()

    def process_query(self, query: str, mock_data: dict) -> dict:
        """
        Executes query retrieval, ranking, context packaging, raw generation simulation, 
        and strict NLI-grounded verification checks.
        
        Args:
            query (str): User query.
            mock_data (dict): Dictionary supplying simulated search database records.
                              Keys: 'sparse_db', 'dense_db'.
            
        Returns:
            dict: Structured response containing verified answers, scores, and telemetry.
        """
        logger.info(f"Processing query: '{query}'")
        
        # Step 1: Execute Hybrid Retrieval (Simulated)
        sparse_hits = mock_data.get("sparse_db", [])
        dense_hits = mock_data.get("dense_db", [])
        
        # Step 2: Unify rankings via Reciprocal Rank Fusion (RRF)
        fused_candidates = compute_rrf(sparse_hits, dense_hits, k=60, top_n=20)
        
        # Step 3: Refine order via Cross-Encoder Reranking
        reranked_candidates = self.reranker.rerank(query, fused_candidates, top_k=8)
        
        # Step 4: Semantic Context Deduplication
        deduplicated = deduplicate_chunks(reranked_candidates, threshold=0.90)
        
        # Step 5: Attention Optimization & Token Budgets packing (Lost-in-the-middle layout)
        context_str, packed_chunks = pack_context(deduplicated, max_tokens=3000)
        
        # Step 6: Grounded Generation (Simulated LLM response matching context evidence)
        # In production, context_str would be injected into the system prompt template
        raw_answer = self._simulate_llm_generation(query, packed_chunks)
        logger.info(f"Simulated LLM Generation: {raw_answer}")
        
        # Step 7: NLI Entailment & Inline Citation Validation
        validation_report = self.citation_validator.validate_answer(raw_answer, packed_chunks)
        
        # Step 8: Refusal Decision Gateway
        final_answer = self.refusal_handler.process_response(
            raw_answer, 
            validation_report, 
            packed_chunks
        )
        
        logger.info(f"Final Grounded Answer: {final_answer}")
        
        return {
            "query": query,
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "is_faithful": validation_report.get("is_faithful", False),
            "faithfulness_score": validation_report.get("faithfulness_score", 0.0),
            "packed_chunks_count": len(packed_chunks),
            "validation_details": validation_report
        }

    def _simulate_llm_generation(self, query: str, context_chunks: list[dict]) -> str:
        """
        Simulates structured grounded generation based on query subject matching context.
        """
        query_lower = query.lower()
        
        if not context_chunks:
            return "INSUFFICIENT_EVIDENCE"
            
        # Simulate retrieval match for AdamW optimizer
        if "optimizer" in query_lower:
            for c in context_chunks:
                if "adamw" in c.get("content", "").lower():
                    return "The architecture was optimized using AdamW with a learning rate of 1e-4 [paper_001:methodology:4]."
            return "The model was trained with SGD [paper_001:methodology:4]." # Intentional incorrect citation
            
        # Simulate retrieval match for validation accuracy
        elif "accuracy" in query_lower:
            for c in context_chunks:
                if "accuracy" in c.get("content", "").lower():
                    return "We achieved a validation accuracy of 92.4% on GLUE benchmarks [paper_001:results:6]."
            return "The model achieved 95% validation accuracy [paper_001:results:6]."
            
        # Standard fallback refusal string
        return "I cannot find supporting evidence within the provided corpus to answer this query."

if __name__ == "__main__":
    pipeline = ResearchRAGPipeline()
    
    # Setup test mock DB payload
    mock_db = {
        "sparse_db": [
            {"chunk_id": "c1", "paper_id": "paper_001", "section": "methodology", "page_number": 4, "content": "The architecture was optimized using AdamW with a learning rate of 1e-4.", "embedding": [0.1] * 1024},
            {"chunk_id": "c2", "paper_id": "paper_001", "section": "results", "page_number": 6, "content": "We achieved a validation accuracy of 92.4% on GLUE benchmarks.", "embedding": [0.2] * 1024}
        ],
        "dense_db": [
            {"chunk_id": "c1", "paper_id": "paper_001", "section": "methodology", "page_number": 4, "content": "The architecture was optimized using AdamW with a learning rate of 1e-4.", "embedding": [0.1] * 1024},
            {"chunk_id": "c2", "paper_id": "paper_001", "section": "results", "page_number": 6, "content": "We achieved a validation accuracy of 92.4% on GLUE benchmarks.", "embedding": [0.2] * 1024}
        ]
    }
    
    # Run test queries
    res_1 = pipeline.process_query("What optimizer was used?", mock_db)
    print("\nQuery 1 Results:")
    print(f"Final Answer: {res_1['final_answer']}")
    print(f"Faithfulness Score: {res_1['faithfulness_score']}")
    
    res_2 = pipeline.process_query("What was the batch size?", mock_db)
    print("\nQuery 2 Results (Expect Refusal):")
    print(f"Final Answer: {res_2['final_answer']}")
