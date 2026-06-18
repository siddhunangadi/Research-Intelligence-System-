"""
Database Index Builder.
Coordinates document parsing ingestion, batch embedding vectorization, 
and atomic transactions to synchronize PostgreSQL metadata and ChromaDB indices.
"""

import logging
from ingestion.ingestion_pipeline import IngestionPipeline
from embeddings.batch_embedding import BatchEmbedder
from indexing.postgres_store import PostgresMetadataStore
from indexing.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(
        self,
        postgres_store: PostgresMetadataStore = None,
        chroma_store: ChromaVectorStore = None,
        batch_embedder: BatchEmbedder = None
    ):
        self.pg_store = postgres_store or PostgresMetadataStore()
        self.chroma_store = chroma_store or ChromaVectorStore()
        self.embedder = batch_embedder or BatchEmbedder()
        self.pipeline = IngestionPipeline()

    def build_index_for_pdf(self, pdf_path: str, title: str = None) -> bool:
        """
        Orchestrates full ingestion, vectorization, and transactional database synchronization.
        
        Args:
            pdf_path (str): Filepath to the PDF document.
            title (str, optional): Overriding title.
            
        Returns:
            bool: Success of synchronized registration.
        """
        try:
            # 1. Parsing & Chunking
            paper_meta, raw_chunks = self.pipeline.ingest_document(pdf_path, title)
            paper_id = paper_meta["paper_id"]
            
            # 2. Batch vectorization
            vectorized_chunks = self.embedder.embed_chunks_batch(raw_chunks)
            
            # 3. Synchronized Database Persistence (Transactional safety)
            # Try PostgreSQL first. If this fails, we do not write to Chroma
            pg_success = self.pg_store.insert_paper(
                paper_id=paper_id,
                title=paper_meta["title"],
                authors=paper_meta["authors"],
                year=paper_meta["year"],
                venue=paper_meta["venue"]
            )
            
            if not pg_success:
                logger.error(f"PostgreSQL paper registration failed for '{paper_id}'. Aborting sync.")
                return False
                
            # Insert chunks metadata into PostgreSQL
            inserted_chunks_count = self.pg_store.insert_chunks_batch(vectorized_chunks)
            if inserted_chunks_count == 0 and len(vectorized_chunks) > 0:
                logger.error("PostgreSQL chunks metadata ingestion failed. Rolling back paper registry.")
                self.pg_store.delete_paper(paper_id)
                return False

            # Compile values for Chroma vector database insertions
            ids = [c["chunk_id"] for c in vectorized_chunks]
            embeddings = [c["embedding"] for c in vectorized_chunks]
            documents = [c["content"] for c in vectorized_chunks]
            
            # We map metadata properties excluding the raw embeddings vectors
            chroma_metadatas = []
            for c in vectorized_chunks:
                meta = {
                    "paper_id": c["paper_id"],
                    "section": c["section"],
                    "page_number": c["page_number"],
                    "token_count": c["token_count"],
                    "embedding_id": c["embedding_id"]
                }
                chroma_metadatas.append(meta)

            # Insert vectors into ChromaDB
            chroma_success = self.chroma_store.add_vectors(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=chroma_metadatas
            )
            
            if not chroma_success:
                # Rollback PostgreSQL registries to maintain absolute database synchronization
                logger.error("ChromaDB vector insertion failed. Rolling back PostgreSQL records.")
                self.pg_store.delete_paper(paper_id)
                return False
                
            logger.info(f"Synchronized registration completed. Paper '{paper_id}' fully indexed in relational and vector stores.")
            return True
            
        except Exception as e:
            logger.error(f"Critical index building failure: {e}")
            return False
            
    def delete_paper_index(self, paper_id: str) -> bool:
        """
        Synchronously removes paper records from both PostgreSQL and ChromaDB stores.
        """
        # Delete from Chroma vector database
        chroma_del = self.chroma_store.delete_by_paper(paper_id)
        # Delete from PostgreSQL metadata database (cascades chunk table entries)
        pg_del = self.pg_store.delete_paper(paper_id)
        
        return chroma_del and pg_del
