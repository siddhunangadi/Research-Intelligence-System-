CREATE TABLE IF NOT EXISTS papers (
    paper_id VARCHAR(50) PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    year INTEGER,
    venue VARCHAR(100),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_year_venue ON papers(year, venue);
