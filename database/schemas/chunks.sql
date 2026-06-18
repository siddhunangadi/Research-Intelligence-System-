CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(50) PRIMARY KEY,
    paper_id VARCHAR(50) REFERENCES papers(paper_id) ON DELETE CASCADE,
    section VARCHAR(50) NOT NULL,
    page_number INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    embedding_id VARCHAR(100) UNIQUE NOT NULL,
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_section ON chunks(paper_id, section);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(embedding_id);
