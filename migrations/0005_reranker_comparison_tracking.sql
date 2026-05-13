-- Migration: Cross-Encoder Comparison Tracking
-- Adds columns to compare chunk ordering before and after cross-encoder reranking.
-- Also backfills rerank_time_ms which was computed but never persisted (bug fix).
ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS rerank_time_ms        INT,
    ADD COLUMN IF NOT EXISTS pre_ce_chunk_ids      INTEGER[],
    ADD COLUMN IF NOT EXISTS post_ce_chunk_ids     INTEGER[];

COMMENT ON COLUMN query_logs.rerank_time_ms    IS 'Time spent in LLM + cross-encoder reranking (ms)';
COMMENT ON COLUMN query_logs.pre_ce_chunk_ids  IS 'Chunk IDs in LLM-ranked order, before cross encoder';
COMMENT ON COLUMN query_logs.post_ce_chunk_ids IS 'Chunk IDs after cross encoder reranking';
