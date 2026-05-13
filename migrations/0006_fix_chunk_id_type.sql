-- transcript_chunks.id is UUID, not INTEGER.
-- Fix the array columns added in 0005 to match the actual type.
ALTER TABLE query_logs
    ALTER COLUMN pre_ce_chunk_ids  TYPE UUID[] USING pre_ce_chunk_ids::text[]::uuid[],
    ALTER COLUMN post_ce_chunk_ids TYPE UUID[] USING post_ce_chunk_ids::text[]::uuid[];
