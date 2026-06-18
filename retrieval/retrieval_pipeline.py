"""
Retrieval Pipeline Orchestrator.
Unifies relational metadata scoping, sparse search indexing, vector similarity searches, 
Rank Fusions, cross-attention re-rankings, deduplication, and context packaging.
"""

import logging
from typing import Optional, List, Dict, Tuple
from configs.config_manager import get_config
from indexing.postgres_store import PostgresMetadataStore, ChunkModel
from indexing.chroma_store import ChromaVectorStore
from retrieval.intent_classifier import IntentClassifier
from retrieval.metadata_filters import MetadataFilterBuilder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.rrf_fusion import compute_rrf
from retrieval.reranker import CrossEncoderReranker
from retrieval.deduplicator import deduplicate_chunks
from retrieval.context_packer import pack_context

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(
        self,
        postgres_store: Optional[PostgresMetadataStore] = None,
        chroma_store: Optional[ChromaVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.config = get_config()
        self.pg_store = postgres_store or PostgresMetadataStore()
        self.chroma_store = chroma_store or ChromaVectorStore()
        
        self.classifier = IntentClassifier()
        self.filter_builder = MetadataFilterBuilder()
        
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(chroma_store=self.chroma_store)
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.retrieval.rerank_model,
            threshold=self.config.retrieval.rerank_threshold
        )

    def _fetch_postgres_chunks(self, paper_id: Optional[str] = None, sections: Optional[List[str]] = None) -> List[Dict]:
        """Queries PostgreSQL metadata database to load raw chunk segments matching scope filters."""
        with self.pg_store.manager.get_session() as session:
            query = session.query(ChunkModel)
            if paper_id:
                query = query.filter(ChunkModel.paper_id == paper_id)
            if sections:
                query = query.filter(ChunkModel.section.in_(sections))
            return [{
                "chunk_id": r.chunk_id, "paper_id": r.paper_id, "section": r.section,
                "page_number": r.page_number, "token_count": r.token_count,
                "content": r.content, "embedding_id": r.embedding_id
            } for r in query.all()]

    def retrieve_grounded_context(
        self, 
        query: str, 
        paper_id: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """Executes end-to-end hybrid semantic search and structures context packing layout."""
        sections = self.classifier.classify_intent(query) or None
            
        for attempt in range(2):
            pg_chunks = self._fetch_postgres_chunks(paper_id, sections)
            
            sparse_hits = []
            if pg_chunks:
                self.bm25.fit(pg_chunks)
                sparse_hits = self.bm25.retrieve(query, top_k=self.config.retrieval.sparse_top_k)
            else:
                logger.warning(f"PostgreSQL/SQLite returned empty chunks for sections {sections}. Skipping BM25 search.")
                
            dense_hits = self.dense.retrieve(
                query=query, 
                top_k=self.config.retrieval.dense_top_k, 
                paper_id=paper_id,
                section=sections[0] if (sections and len(sections) == 1) else None
            )
            
            fused_candidates = compute_rrf(
                sparse_results=sparse_hits, 
                dense_results=dense_hits, 
                k=self.config.retrieval.rrf_k,
                top_n=25
            )
            
            # If we got no results, but we filtered by section, retry without the section constraint
            if not fused_candidates and sections is not None:
                logger.info(f"Retrieval returned 0 candidates with section filters: {sections}. Retrying with global search.")
                sections = None
            else:
                break
        
        reranked = self.reranker.rerank(
            query=query, 
            chunks=fused_candidates, 
            top_k=self.config.retrieval.rerank_top_k
        )
        
        deduplicated = deduplicate_chunks(
            chunks=reranked, 
            threshold=self.config.context.deduplication_threshold
        )
        
        context_str, packed_chunks = pack_context(
            chunks=deduplicated,
            max_tokens=self.config.context.max_tokens
        )
        
        logger.info(f"Retrieval pipeline complete. Packed {len(packed_chunks)} chunks for answer generation.")
        return context_str, packed_chunks
