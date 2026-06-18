"""
ChromaDB Vector Store interface.
Manages connection initialization, collection configuration, 
batch vector insertion, and structured metadata filter search queries.
"""

import logging
import chromadb
from chromadb.config import Settings
from configs.config_manager import get_config

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, collection_name: str = "scientific_papers"):
        self.config = get_config()
        self.path = self.config.database.chroma_path
        self.collection_name = collection_name
        
        try:
            logger.info(f"Connecting to ChromaDB at: {self.path}")
            # Initialize persistent local SQLite client for Chroma
            self.client = chromadb.PersistentClient(
                path=self.path,
                settings=Settings(allow_reset=True)
            )
            # Create or get existing index collection configured with Cosine distance metric
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Vector collection '{self.collection_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Critical error initializing ChromaDB: {e}")
            raise

    def add_vectors(
        self, 
        ids: list[str], 
        embeddings: list[list[float]], 
        documents: list[str], 
        metadatas: list[dict]
    ) -> bool:
        """
        Inserts vector embeddings, text content, and metadata parameters into the collection.
        """
        if not ids:
            return False
            
        try:
            # ChromaDB natively handles bulk insertions
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(ids)} vectors to Chroma collection '{self.collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"ChromaDB insert failed: {e}")
            return False

    def query_similarity(
        self, 
        query_vector: list[float], 
        top_k: int = 10, 
        where_filter: dict = None
    ) -> list[dict]:
        """
        Queries Chroma for vector nearest neighbors with matching metadata criteria.
        
        Args:
            query_vector (list[float]): Asymmetric search embedding vector.
            top_k (int): Number of neighbors to return.
            where_filter (dict, optional): Metadata logic filters. E.g. {"paper_id": "p1"}
            
        Returns:
            list[dict]: Unified lists containing chunk_id, content, metadata and similarity distance.
        """
        try:
            # Query the vector collection
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if not results or not results["ids"] or not results["ids"][0]:
                return []
                
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            # Chroma returns distances (dissimilarity). We compute similarity as 1 - distance.
            distances = results["distances"][0]
            
            for idx in range(len(ids)):
                sim_score = 1.0 - distances[idx]
                chunk_data = {
                    "chunk_id": ids[idx],
                    "content": docs[idx],
                    "similarity_score": float(sim_score)
                }
                # Merge metadata properties directly
                if metas[idx]:
                    chunk_data.update(metas[idx])
                formatted_results.append(chunk_data)
                
            return formatted_results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def delete_by_paper(self, paper_id: str) -> bool:
        """
        Removes all chunk vectors matching a specific paper ID.
        """
        try:
            self.collection.delete(where={"paper_id": paper_id})
            logger.info(f"Removed all vectors matching paper ID '{paper_id}' from vector db.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors for paper '{paper_id}': {e}")
            return False

    def reset_database(self):
        """
        Resets vector databases indexes completely.
        """
        try:
            self.client.reset()
            logger.warning("ChromaDB vector registry has been reset to empty state.")
            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Reset database failed: {e}")
            raise
