"""
PostgreSQL Metadata Store.
Implements SQLAlchemy schemas for scientific papers and semantic text chunks, 
and handles repository actions such as bulk inserts and status checking.
"""

import logging
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, Float, Boolean
from sqlalchemy.orm import relationship, Session
from database.connection import Base, DatabaseConnectionManager

logger = logging.getLogger(__name__)

# SQLAlchemy Relational Models mapping database schemas
class PaperModel(Base):
    __tablename__ = "papers"
    
    paper_id = Column(String(50), primary_key=True)
    title = Column(Text, nullable=False)
    authors = Column(Text)
    year = Column(Integer)
    venue = Column(String(100))
    ingested_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("ChunkModel", back_populates="paper", cascade="all, delete-orphan")

class ChunkModel(Base):
    __tablename__ = "chunks"
    
    chunk_id = Column(String(50), primary_key=True)
    paper_id = Column(String(50), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False)
    section = Column(String(50), nullable=False)
    page_number = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding_id = Column(String(100), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    
    paper = relationship("PaperModel", back_populates="chunks")

class EvaluationModel(Base):
    __tablename__ = "evaluations"
    
    evaluation_id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    commit_hash = Column(String(40), nullable=True)
    faithfulness = Column(Float, nullable=False)
    answer_relevancy = Column(Float, nullable=False)
    context_precision = Column(Float, nullable=False)
    context_recall = Column(Float, nullable=False)
    retrieval_mrr = Column(Float, nullable=True)
    retrieval_recall_at_5 = Column(Float, nullable=True)
    passed_gate = Column(Boolean, nullable=False)

class QueryLogModel(Base):
    __tablename__ = "query_logs"
    
    query_id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    latency_ms = Column(Float)
    token_count = Column(Integer)
    refusal_triggered = Column(Boolean)
    nli_passed = Column(Boolean)


class PostgresMetadataStore:
    def __init__(self, connection_manager: DatabaseConnectionManager = None):
        self.manager = connection_manager or DatabaseConnectionManager()
        # Ensure tables are generated if they do not exist
        try:
            Base.metadata.create_all(self.manager.engine)
            logger.info("Relational database tables checked/created successfully.")
        except Exception as e:
            logger.error(f"Error creating database tables in postgres_store: {e}")

    def insert_paper(self, paper_id: str, title: str, authors: str = None, year: int = None, venue: str = None) -> bool:
        """
        Inserts or updates a scientific paper registry record.
        """
        with self.manager.get_session() as session:
            existing = session.query(PaperModel).filter(PaperModel.paper_id == paper_id).first()
            if existing:
                logger.info(f"Paper '{paper_id}' already registered. Updating metadata.")
                existing.title = title
                existing.authors = authors
                existing.year = year
                existing.venue = venue
            else:
                new_paper = PaperModel(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue
                )
                session.add(new_paper)
                logger.info(f"Registered new paper: {paper_id}")
            return True

    def insert_chunks_batch(self, chunks: list[dict]) -> int:
        """
        Inserts a list of chunks in bulk transaction block for high ingestion throughput.
        
        Args:
            chunks (list[dict]): List of dictionaries representing mapped chunks.
                                 Must contain keys: chunk_id, paper_id, section, page_number,
                                 token_count, embedding_id, content.
        """
        if not chunks:
            return 0
            
        with self.manager.get_session() as session:
            count = 0
            for chunk_data in chunks:
                chunk_id = chunk_data["chunk_id"]
                # Prevent unique constraint violations
                existing = session.query(ChunkModel).filter(ChunkModel.chunk_id == chunk_id).first()
                if existing:
                    existing.section = chunk_data["section"]
                    existing.page_number = chunk_data["page_number"]
                    existing.token_count = chunk_data["token_count"]
                    existing.content = chunk_data["content"]
                    existing.embedding_id = chunk_data["embedding_id"]
                else:
                    new_chunk = ChunkModel(
                        chunk_id=chunk_id,
                        paper_id=chunk_data["paper_id"],
                        section=chunk_data["section"],
                        page_number=chunk_data["page_number"],
                        token_count=chunk_data["token_count"],
                        embedding_id=chunk_data["embedding_id"],
                        content=chunk_data["content"]
                    )
                    session.add(new_chunk)
                count += 1
            logger.info(f"Successfully batch-inserted/updated {count} chunks in PostgreSQL.")
            return count

    def get_paper_metadata(self, paper_id: str) -> dict:
        """
        Retrieves registration detail of a registered paper.
        """
        with self.manager.get_session() as session:
            paper = session.query(PaperModel).filter(PaperModel.paper_id == paper_id).first()
            if not paper:
                return {}
            return {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "venue": paper.venue,
                "ingested_at": paper.ingested_at.isoformat() if paper.ingested_at else None
            }

    def list_all_papers(self) -> list[dict]:
        """
        Lists all registered papers.
        """
        with self.manager.get_session() as session:
            papers = session.query(PaperModel).all()
            return [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "venue": p.venue
                } for p in papers
            ]
            
    def delete_paper(self, paper_id: str) -> bool:
        """
        Deletes a paper registry record (triggers cascade delete on child chunks).
        """
        with self.manager.get_session() as session:
            paper = session.query(PaperModel).filter(PaperModel.paper_id == paper_id).first()
            if paper:
                session.delete(paper)
                logger.info(f"Deleted paper registry '{paper_id}' and all matching chunks.")
                return True
            return False

    def insert_query_log(self, query_id: str, query_text: str, response_text: str, latency_ms: float, token_count: int, refusal_triggered: bool, nli_passed: bool) -> bool:
        """
        Inserts query telemetry log.
        """
        try:
            with self.manager.get_session() as session:
                log_record = QueryLogModel(
                    query_id=query_id,
                    query_text=query_text,
                    response_text=response_text,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    refusal_triggered=refusal_triggered,
                    nli_passed=nli_passed
                )
                session.add(log_record)
            return True
        except Exception as e:
            logger.error(f"Failed to log query telemetry to postgres: {e}")
            return False

    def list_query_logs(self) -> list[dict]:
        """
        Lists all query telemetry logs.
        """
        try:
            with self.manager.get_session() as session:
                logs = session.query(QueryLogModel).order_by(QueryLogModel.timestamp.desc()).all()
                return [
                    {
                        "query_id": l.query_id,
                        "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                        "query_text": l.query_text,
                        "response_text": l.response_text,
                        "latency_ms": l.latency_ms,
                        "token_count": l.token_count,
                        "refusal_triggered": l.refusal_triggered,
                        "nli_passed": l.nli_passed
                    } for l in logs
                ]
        except Exception as e:
            logger.error(f"Failed to list query logs: {e}")
            return []

