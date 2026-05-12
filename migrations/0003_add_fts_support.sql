-- 1. Add a searchable tsvector column
-- This column automatically pre-tokensizes the content for fast searching
ALTER TABLE transcript_chunks
ADD COLUMN IF NOT EXISTS fts_tokens tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- 2. Add a GIN (Generalized Inverted Index)
-- This is what makes keyword search O(log N) instead of a full table scan
CREATE INDEX IF NOT EXISTS idx_fts_content ON transcript_chunks USING GIN(fts_tokens);
