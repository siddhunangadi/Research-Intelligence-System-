"""
Dense Semantic Vector Retriever.
Integrates vector embedder and ChromaDB persistent search index 
to execute similarity nearest-neighbors lookups.
"""

import logging
from typing import Optional, List, Dict
from embeddings.embedder import ScientificEmbedder
from indexing.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(
        self, 
        chroma_store: Optional[ChromaVectorStore] = None, 
        embedder: Optional[ScientificEmbedder] = None
    ):
        self.store = chroma_store or ChromaVectorStore()
        self.embedder = embedder or ScientificEmbedder()

    def _mock_retrieve(self, query: str, top_k: int, paper_id: Optional[str], section: Optional[str]) -> List[Dict]:
        """Simulates dense retrieval via keyword overlap when model is None."""
        logger.info("Embedder running in mock mode. Simulating dense retrieval via keyword overlap.")
        from indexing.postgres_store import PostgresMetadataStore, ChunkModel
        pg_store = PostgresMetadataStore()
        
        with pg_store.manager.get_session() as session:
            db_query = session.query(ChunkModel)
            if paper_id:
                db_query = db_query.filter(ChunkModel.paper_id == paper_id)
            if section:
                db_query = db_query.filter(ChunkModel.section == section)
            chunks = db_query.all()
            mapped = [{
                "chunk_id": r.chunk_id, "paper_id": r.paper_id, "section": r.section,
                "page_number": r.page_number, "token_count": r.token_count,
                "content": r.content, "embedding_id": r.embedding_id
            } for r in chunks]

        stop_words = {"what", "was", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "by", "on", "at", "is", "are"}
        query_words = set(query.lower().split()) - stop_words
        
        scored = []
        for r in mapped:
            overlap = sum(1 for w in query_words if w in r["content"].lower())
            if overlap > 0:
                scored.append((overlap, r))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [{**r, "dense_score": float(0.5 + min(o * 0.1, 0.45))} for o, r in scored[:top_k]]
        
        if not results and mapped:
            results = [{**r, "dense_score": 0.5} for r in mapped[:top_k]]
            
        logger.info(f"Mock dense retriever returned {len(results)} candidates.")
        return results

    def retrieve(
        self, 
        query: str, 
        top_k: int = 20, 
        paper_id: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[Dict]:
        """Executes dense similarity search over ChromaDB index."""
        if self.embedder.model is None:
            return self._mock_retrieve(query, top_k, paper_id, section)

        query_vector = self.embedder.embed_query(query)
        
        where_filter = {}
        if paper_id:
            where_filter["paper_id"] = paper_id
        if section:
            where_filter["section"] = section
            
        if len(where_filter) > 1:
            where_filter = {"$and": [{k: v} for k, v in where_filter.items()]}
        elif not where_filter:
            where_filter = None

        logger.debug(f"Executing Chroma dense query with filters: {where_filter}")
        results = self.store.query_similarity(
            query_vector=query_vector,
            top_k=top_k,
            where_filter=where_filter
        )
        
        dense_results = [{**res, "dense_score": res.get("similarity_score", 0.0)} for res in results]
        logger.info(f"Dense retriever returned {len(dense_results)} candidates.")
        return dense_results
