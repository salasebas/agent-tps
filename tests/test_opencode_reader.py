import json
from pathlib import Path
import sqlite3

import pytest

from tokpulse.providers.opencode import OpenCodeDBReader


@pytest.fixture
def mock_opencode_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "opencode.db"
    conn = sqlite3.connect(db_file)
    with conn:
        conn.executescript("""
            CREATE TABLE project (
                id text PRIMARY KEY
            );
            CREATE TABLE session (
                id text PRIMARY KEY,
                project_id text NOT NULL,
                parent_id text,
                slug text NOT NULL,
                directory text NOT NULL,
                title text NOT NULL,
                version text NOT NULL,
                share_url text,
                summary_additions integer,
                summary_deletions integer,
                summary_files integer,
                summary_diffs text,
                revert text,
                permission text,
                time_created integer NOT NULL,
                time_updated integer NOT NULL,
                time_compacting integer,
                time_archived integer,
                workspace_id text,
                path text,
                agent text,
                model text,
                cost real DEFAULT 0 NOT NULL,
                tokens_input integer DEFAULT 0 NOT NULL,
                tokens_output integer DEFAULT 0 NOT NULL,
                tokens_reasoning integer DEFAULT 0 NOT NULL,
                tokens_cache_read integer DEFAULT 0 NOT NULL,
                tokens_cache_write integer DEFAULT 0 NOT NULL,
                metadata text
            );
            CREATE TABLE message (
                id text PRIMARY KEY,
                session_id text NOT NULL,
                time_created integer NOT NULL,
                time_updated integer NOT NULL,
                data text NOT NULL
            );
            CREATE TABLE part (
                id text PRIMARY KEY,
                message_id text NOT NULL,
                session_id text NOT NULL,
                time_created integer NOT NULL,
                time_updated integer NOT NULL,
                data text NOT NULL
            );
        """)
        # Insert project
        conn.execute("INSERT INTO project (id) VALUES ('proj_1')")
        # Insert session
        model_json = json.dumps({"id": "longcat-2.5", "providerID": "opencode"})
        conn.execute(
            """
            INSERT INTO session (
                id, project_id, slug, directory, title, version,
                time_created, time_updated, agent, model, cost,
                tokens_input, tokens_output, tokens_reasoning,
                tokens_cache_read, tokens_cache_write
            ) VALUES (
                'ses_mock_1', 'proj_1', 'mock-slug', '/tmp', 'Test Session', '1.0',
                1700000000000, 1700000005000, 'build', ?, 0.002,
                1000, 200, 50, 500, 0
            )
        """,
            (model_json,),
        )

        # Insert message
        msg_data = json.dumps(
            {
                "role": "assistant",
                "time": {"created": 1700000001000, "completed": 1700000004500},
            }
        )
        conn.execute(
            """
            INSERT INTO message (id, session_id, time_created, time_updated, data)
            VALUES ('msg_1', 'ses_mock_1', 1700000001000, 1700000004500, ?)
        """,
            (msg_data,),
        )

        # Insert parts with text timing
        part_data = json.dumps(
            {
                "type": "text",
                "text": "Hello world response",
                "time": {"start": 1700000001500, "end": 1700000004000},
            }
        )
        conn.execute(
            """
            INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
            VALUES ('prt_1', 'msg_1', 'ses_mock_1', 1700000001500, 1700000004000, ?)
        """,
            (part_data,),
        )

    conn.close()
    return db_file


def test_opencode_reader_parses_session(mock_opencode_db: Path):
    reader = OpenCodeDBReader(db_path=mock_opencode_db)
    assert reader.is_available()

    sessions = reader.get_recent_sessions(limit=5)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "ses_mock_1"
    assert s.title == "Test Session"
    assert s.provider == "opencode"
    assert s.model == "longcat-2.5"
    assert s.tokens.input_tokens == 1000
    assert s.tokens.output_tokens == 200
    assert s.tokens.reasoning_tokens == 50
    assert s.tokens.cached_read_tokens == 500
    # cache hit rate: 500 / 1500 = 0.333
    assert abs(s.tokens.cache_hit_rate - (500 / 1500)) < 1e-3

    # check timing from parts
    assert s.timings.generation_duration_ms == 2500.0  # 4000 - 1500
    # generated = 200 + 50 = 250 tokens in 2.5s -> decode_tps = 100 tok/s
    assert s.tps.decode_tps == 100.0


def test_opencode_reader_real_db():
    reader = OpenCodeDBReader()
    if reader.is_available():
        sessions = reader.get_recent_sessions(limit=3)
        assert len(sessions) > 0
        for s in sessions:
            assert s.session_id.startswith("ses_")
            assert s.tps.decode_tps >= 0.0
