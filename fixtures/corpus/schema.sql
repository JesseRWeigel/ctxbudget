CREATE TABLE context_run (
    run_id          BIGSERIAL PRIMARY KEY,
    model_name      TEXT        NOT NULL,
    window_tokens   INTEGER     NOT NULL CHECK (window_tokens > 0),
    reserved_output INTEGER     NOT NULL CHECK (reserved_output >= 0),
    input_tokens    INTEGER     NOT NULL CHECK (input_tokens >= 0),
    counted_exactly BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT reserved_fits CHECK (reserved_output < window_tokens)
);

CREATE INDEX context_run_model_created_idx ON context_run (model_name, created_at DESC);

CREATE TABLE context_part (
    part_id     BIGSERIAL PRIMARY KEY,
    run_id      BIGINT  NOT NULL REFERENCES context_run (run_id) ON DELETE CASCADE,
    label       TEXT    NOT NULL,
    kind        TEXT    NOT NULL CHECK (kind IN ('system', 'file', 'overhead')),
    tokens      INTEGER NOT NULL CHECK (tokens >= 0),
    cut_rank    INTEGER,
    cut_reason  TEXT
);

CREATE VIEW context_headroom AS
SELECT r.run_id,
       r.model_name,
       r.window_tokens - r.input_tokens                           AS headroom_tokens,
       LEAST(r.reserved_output, r.window_tokens - r.input_tokens) AS usable_reply_tokens
FROM   context_run r;

INSERT INTO context_run (model_name, window_tokens, reserved_output, input_tokens, counted_exactly)
VALUES ('gpt-4o', 128000, 16384, 94112, TRUE),
       ('qwen2.5-7b-instruct', 32768, 4096, 30100, FALSE);
