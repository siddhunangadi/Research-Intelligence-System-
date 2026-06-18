"""
FastAPI semantic query search router.
Orchestrates hybrid search, answer completion, NLI validations, and refusal checks.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from retrieval.retrieval_pipeline import RetrievalPipeline
from generation.answer_generator import AnswerGenerator
from validation.citation_validator import CitationValidator
from validation.refusal_handler import RefusalHandler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["query"])

class QueryRequest(BaseModel):
    query: str
    paper_id: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    is_faithful: bool
    faithfulness_score: float
    sources: list[dict]
    verified_claims: list[dict]
    unverified_claims: list[dict]

# Instantiated pipelines
pipeline = RetrievalPipeline()
generator = AnswerGenerator()
validator = CitationValidator()
refusal_handler = RefusalHandler()

@router.post("", response_model=QueryResponse)
def execute_rag_search(request: QueryRequest):
    """
    Executes hybrid retrieval, grounded generation, and strict citation validations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        # 1. Hybrid Retrieval & Context Packing
        context_str, retrieved_chunks = pipeline.retrieve_grounded_context(
            query=request.query,
            paper_id=request.paper_id
        )
        
        # 2. Inquire grounded LLM completion
        raw_answer = generator.generate_answer(
            query=request.query,
            context_str=context_str
        )
        
        # 3. Entailment checks and citation audits
        validation_report = validator.validate_answer(
            answer=raw_answer,
            retrieved_chunks=retrieved_chunks
        )
        
        # 4. Enforce refusal gate
        final_answer = refusal_handler.process_response(
            raw_answer=raw_answer,
            validation_results=validation_report,
            retrieved_chunks=retrieved_chunks
        )
        
        # Clean chunks metadata properties to prevent serialization exceptions
        serializable_chunks = []
        for c in retrieved_chunks:
            chunk_item = dict(c)
            # Remove vector weights from return mappings
            if "embedding" in chunk_item:
                del chunk_item["embedding"]
            serializable_chunks.append(chunk_item)

        # Unify reports
        response_data = QueryResponse(
            query=request.query,
            answer=final_answer,
            is_faithful=validation_report.get("is_faithful", False),
            faithfulness_score=validation_report.get("faithfulness_score", 0.0),
            sources=serializable_chunks,
            verified_claims=validation_report.get("verified_claims", []),
            unverified_claims=validation_report.get("unverified_claims", [])
        )
        
        logger.info(f"Query search resolved. Faithfulness score: {response_data.faithfulness_score:.2f}")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to execute search pipeline for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")
