"""数据库表结构 DDL（幂等）。

直接用 psycopg 执行；服务启动时在 lifespan 中检查/初始化。
不使用 ORM / 迁移框架，备份与迁移走 shell 全量备份（见 deploy/backup.sh）。
逐条执行（psycopg 扩展协议不支持单次多语句）。
"""

from __future__ import annotations

SCHEMA_STATEMENTS: list[str] = [
    # ── 对话（消息树根） ────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id                     text PRIMARY KEY,
        owner_id               text NOT NULL,
        org_id                 text,
        visibility             text NOT NULL DEFAULT 'private',
        title                  text NOT NULL DEFAULT '',
        active_leaf_message_id text,
        session_id             text,
        forked_from            text,
        created_at             timestamptz NOT NULL DEFAULT now(),
        updated_at             timestamptz NOT NULL DEFAULT now()
    )
    """,
    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS forked_from text",
    "CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_org ON conversations(org_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_visibility ON conversations(visibility)",
    # ── 消息（树节点；parent_id 自引用，分叉点多子） ────────
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              text PRIMARY KEY,
        conversation_id text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        parent_id       text REFERENCES messages(id) ON DELETE CASCADE,
        role            text NOT NULL,
        content         text NOT NULL DEFAULT '',
        hits            jsonb,
        context         text,
        research        jsonb,
        usage           jsonb,
        latency_s       double precision,
        params          jsonb,
        status          text NOT NULL DEFAULT 'done',
        error           text,
        created_at      timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)",
    # ── 生成运行（生产级异步 run） ───────────────────────────
    """
    CREATE TABLE IF NOT EXISTS generation_runs (
        id                   text PRIMARY KEY,
        conversation_id      text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        user_message_id      text NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        assistant_message_id text NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        owner_id             text NOT NULL,
        org_id               text,
        status               text NOT NULL DEFAULT 'queued',
        params               jsonb,
        error                text,
        cancel_requested     boolean NOT NULL DEFAULT false,
        redis_stream         text,
        artifact_prefix      text,
        created_at           timestamptz NOT NULL DEFAULT now(),
        updated_at           timestamptz NOT NULL DEFAULT now(),
        started_at           timestamptz,
        finished_at          timestamptz
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_generation_runs_conv ON generation_runs(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_generation_runs_assistant ON generation_runs(assistant_message_id)",
    "CREATE INDEX IF NOT EXISTS idx_generation_runs_owner ON generation_runs(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_generation_runs_status ON generation_runs(status)",
    # ── 生成事件（持久化 SSE 回放） ─────────────────────────
    """
    CREATE TABLE IF NOT EXISTS message_events (
        id         bigserial PRIMARY KEY,
        run_id     text NOT NULL REFERENCES generation_runs(id) ON DELETE CASCADE,
        seq        bigint NOT NULL,
        type       text NOT NULL,
        payload    jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (run_id, seq)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_message_events_run_seq ON message_events(run_id, seq)",
    # ── 文献库归属与可见性 ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS kb_collections (
        name         text PRIMARY KEY,
        display_name text NOT NULL DEFAULT '',
        owner_id     text NOT NULL,
        org_id       text,
        visibility   text NOT NULL DEFAULT 'private',
        created_at   timestamptz NOT NULL DEFAULT now(),
        updated_at   timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kb_owner ON kb_collections(owner_id)",
    # ── 文献条目 ────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS documents (
        id              text PRIMARY KEY,
        collection_name text NOT NULL REFERENCES kb_collections(name) ON DELETE CASCADE,
        owner_id        text NOT NULL,
        doc_id          text NOT NULL,
        title           text,
        filename        text,
        year            integer,
        pdf_object_key  text,
        artifact_prefix text,
        source_document_id text,
        status          text NOT NULL DEFAULT 'parsing',
        task_id         text,
        chunk_count     integer NOT NULL DEFAULT 0,
        created_at      timestamptz NOT NULL DEFAULT now(),
        updated_at      timestamptz NOT NULL DEFAULT now()
    )
    """,
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_object_key text",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS artifact_prefix text",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_document_id text",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_collection_doc_id ON documents(collection_name, doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_documents_coll ON documents(collection_name)",
    "CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)",
    # ── 自定义 skill 归属与可见性 ───────────────────────────
    """
    CREATE TABLE IF NOT EXISTS user_skills (
        owner_id    text NOT NULL,
        id          text NOT NULL,
        org_id      text,
        visibility  text NOT NULL DEFAULT 'private',
        name        text NOT NULL DEFAULT '',
        description text,
        source_owner_id text,
        source_skill_id text,
        created_at  timestamptz NOT NULL DEFAULT now(),
        updated_at  timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (owner_id, id)
    )
    """,
    "ALTER TABLE user_skills ADD COLUMN IF NOT EXISTS source_owner_id text",
    "ALTER TABLE user_skills ADD COLUMN IF NOT EXISTS source_skill_id text",
    "CREATE INDEX IF NOT EXISTS idx_user_skills_visibility ON user_skills(visibility)",
    "CREATE INDEX IF NOT EXISTS idx_user_skills_org ON user_skills(org_id)",
    # ── 对话分享链接（不透明 token；撤销后 token 失效） ─────────
    """
    CREATE TABLE IF NOT EXISTS conversation_shares (
        token           text PRIMARY KEY,
        conversation_id text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        owner_id        text NOT NULL,
        created_at      timestamptz NOT NULL DEFAULT now(),
        revoked_at      timestamptz
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conversation_shares_conv ON conversation_shares(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_shares_owner ON conversation_shares(owner_id)",
]
