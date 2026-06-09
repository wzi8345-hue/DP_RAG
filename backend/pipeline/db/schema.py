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
        created_at             timestamptz NOT NULL DEFAULT now(),
        updated_at             timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_org ON conversations(org_id)",
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
        status          text NOT NULL DEFAULT 'parsing',
        task_id         text,
        chunk_count     integer NOT NULL DEFAULT 0,
        created_at      timestamptz NOT NULL DEFAULT now(),
        updated_at      timestamptz NOT NULL DEFAULT now()
    )
    """,
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
        created_at  timestamptz NOT NULL DEFAULT now(),
        updated_at  timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (owner_id, id)
    )
    """,
]
