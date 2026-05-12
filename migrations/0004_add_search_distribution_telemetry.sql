-- Adding columns to track the retrieval "consensus"
ALTER TABLE query_logs
ADD COLUMN IF NOT EXISTS search_overlap_count INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS vector_signal INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS keyword_signal INT DEFAULT 0;

-- Optional: Add a comment to describe the columns for future-you
COMMENT ON COLUMN query_logs.search_overlap_count IS 'Number of chunks found by both Vector and Keyword engines';
COMMENT ON COLUMN query_logs.vector_signal IS 'Number of chunks found ONLY by the Vector engine';
COMMENT ON COLUMN query_logs.keyword_signal IS 'Number of chunks found ONLY by the Keyword engine';
