"""
PostgreSQL connection pool and SQLAlchemy session factory.
Manages persistent connections and session lifetimes safely using context managers.
"""

import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from configs.config_manager import get_config

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy declarative models
Base = declarative_base()

class DatabaseConnectionManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseConnectionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        config = get_config()
        self.db_url = config.database.postgres_url
        
        try:
            if self.db_url.startswith("sqlite"):
                logger.info("Configuring SQLite database engine...")
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False},
                    echo=False
                )
            else:
                logger.info("Initializing PostgreSQL database engine pool...")
                # Configure engine with connection pooling parameters for production durability
                self.engine = create_engine(
                    self.db_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=1800,  # Recycle connections after 30 minutes
                    pool_pre_ping=True,  # Check connections before checkout to prevent broken pipe errors
                    echo=False           # Set to True to log SQL statements for debugging
                )
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load database driver or connect ({e}). Falling back to SQLite file database at data/metadata.db.")
            import os
            os.makedirs("data", exist_ok=True)
            self.engine = create_engine(
                "sqlite:///data/metadata.db",
                connect_args={"check_same_thread": False},
                echo=False
            )



            
        try:
            self.SessionFactory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            self._initialized = True
            logger.info("Database connection factory initialized successfully.")
        except Exception as e:
            logger.error(f"Critical error initializing connection factory: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager that yields a transactional session and automatically
        handles commits, rollbacks, and clean-up.
        """
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction error. Session rolled back: {e}")
            raise
        finally:
            session.close()

# Helper function for Dependency Injection
def get_db_session() -> Generator[Session, None, None]:
    manager = DatabaseConnectionManager()
    with manager.get_session() as session:
        yield session
