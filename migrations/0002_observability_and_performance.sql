-- Migration: Phase 4 Observability & Performance
-- Description: Create logging table with split-latency tracking and metadata indices.

CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    refined_query TEXT,
    company_filter TEXT,
    fy_filter TEXT,
    quarter_filter TEXT,
    top_distance FLOAT,

    -- Split Latency Tracking (ms)
    processing_time_ms INT,  -- Entity extraction / Routing
    retrieval_time_ms INT,   -- Vector search
    synthesis_time_ms INT,   -- LLM Answer generation
    response_time_ms INT,    -- Total E2E time

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indices for RAG Filtering performance
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_filters ON transcript_chunks (company, fy, quarter);

-- Indices for Stats Dashboard performance
CREATE INDEX IF NOT EXISTS idx_logs_company ON query_logs(company_filter);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON query_logs(created_at);
