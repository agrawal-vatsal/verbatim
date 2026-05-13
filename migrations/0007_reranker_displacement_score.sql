-- Tracks how much the LLM reranker displaced chunks from the original RRF order.
-- 0 = top-K returned unchanged, 1 = bottom-K promoted to the top.
ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS reranker_displacement FLOAT;

COMMENT ON COLUMN query_logs.reranker_displacement IS '0 = no displacement (top-K returned as-is), 1 = maximum displacement (bottom-K promoted)';
