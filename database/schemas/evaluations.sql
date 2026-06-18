CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    commit_hash VARCHAR(40),
    faithfulness DOUBLE PRECISION NOT NULL,
    answer_relevancy DOUBLE PRECISION NOT NULL,
    context_precision DOUBLE PRECISION NOT NULL,
    context_recall DOUBLE PRECISION NOT NULL,
    retrieval_mrr DOUBLE PRECISION,
    retrieval_recall_at_5 DOUBLE PRECISION,
    passed_gate BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp);
