"""Postgres repositories for conversations, resources, sharing and visibility."""

from __future__ import annotations

import secrets
import uuid
from typing import Any

from psycopg.types.json import Jsonb

from ..auth import AuthContext
from . import base
from .models import (
    Conversation,
    ConversationShare,
    Document,
    GenerationRun,
    IngestTask,
    IngestTaskEvent,
    IngestTaskItem,
    IngestItemStatus,
    IngestTaskStatus,
    KbCollection,
    Message,
    MessageEvent,
    RunStatus,
    UserSkill,
    Visibility,
)


def _model(model, row):
    return model.model_validate(row) if row else None


def _json(value: Any) -> Jsonb | None:
    return Jsonb(value) if value is not None else None


def available() -> bool:
    return base.configured()


# ---------------------------------------------------------------------------
# Collections / documents
# ---------------------------------------------------------------------------


def upsert_collection(
    *,
    name: str,
    display_name: str,
    auth: AuthContext,
    visibility: Visibility = "private",
) -> KbCollection:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO kb_collections(name, display_name, owner_id, org_id, visibility, updated_at)
            VALUES (%(name)s, %(display_name)s, %(owner_id)s, %(org_id)s, %(visibility)s, now())
            ON CONFLICT (name) DO UPDATE SET
              display_name = COALESCE(NULLIF(EXCLUDED.display_name, ''), kb_collections.display_name),
              updated_at = now()
            RETURNING *
            """,
            {
                "name": name,
                "display_name": display_name,
                "owner_id": auth.user_id,
                "org_id": auth.org_id,
                "visibility": visibility,
            },
        )
        return KbCollection.model_validate(cur.fetchone())


def get_collection(name: str) -> KbCollection | None:
    with base.cursor() as cur:
        cur.execute("SELECT * FROM kb_collections WHERE name = %s", (name,))
        return _model(KbCollection, cur.fetchone())


def list_collections(auth: AuthContext) -> list[KbCollection]:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM kb_collections
            WHERE owner_id = %(user_id)s
               OR visibility = 'public'
               OR (visibility = 'org' AND org_id IS NOT NULL AND org_id = %(org_id)s)
            ORDER BY
              CASE
                WHEN visibility = 'public' THEN 0
                WHEN visibility = 'org' THEN 1
                WHEN owner_id = %(user_id)s THEN 2
                ELSE 3
              END,
              updated_at DESC
            """,
            {"user_id": auth.user_id, "org_id": auth.org_id},
        )
        return [KbCollection.model_validate(r) for r in cur.fetchall()]


def update_collection_visibility(name: str, visibility: Visibility, auth: AuthContext) -> KbCollection | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE kb_collections
            SET visibility = %(visibility)s, updated_at = now()
            WHERE name = %(name)s AND owner_id = %(owner_id)s
            RETURNING *
            """,
            {"name": name, "visibility": visibility, "owner_id": auth.user_id},
        )
        return _model(KbCollection, cur.fetchone())


def delete_collection_metadata(name: str, auth: AuthContext) -> bool:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM kb_collections WHERE name = %s AND owner_id = %s",
            (name, auth.user_id),
        )
        return cur.rowcount > 0


def upsert_document(
    *,
    collection_name: str,
    doc_id: str,
    owner_id: str,
    title: str | None = None,
    filename: str | None = None,
    year: int | None = None,
    pdf_object_key: str | None = None,
    artifact_prefix: str | None = None,
    status: str = "parsing",
    task_id: str | None = None,
    chunk_count: int = 0,
    source_document_id: str | None = None,
) -> Document:
    doc_pk = f"{collection_name}:{doc_id}"
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents(
              id, collection_name, owner_id, doc_id, title, filename, year,
              pdf_object_key, artifact_prefix, source_document_id, status, task_id,
              chunk_count, updated_at
            )
            VALUES (
              %(id)s, %(collection_name)s, %(owner_id)s, %(doc_id)s, %(title)s, %(filename)s,
              %(year)s, %(pdf_object_key)s, %(artifact_prefix)s, %(source_document_id)s,
              %(status)s, %(task_id)s, %(chunk_count)s, now()
            )
            ON CONFLICT (collection_name, doc_id) DO UPDATE SET
              title = COALESCE(EXCLUDED.title, documents.title),
              filename = COALESCE(EXCLUDED.filename, documents.filename),
              year = COALESCE(EXCLUDED.year, documents.year),
              pdf_object_key = COALESCE(EXCLUDED.pdf_object_key, documents.pdf_object_key),
              artifact_prefix = COALESCE(EXCLUDED.artifact_prefix, documents.artifact_prefix),
              status = EXCLUDED.status,
              task_id = COALESCE(EXCLUDED.task_id, documents.task_id),
              chunk_count = GREATEST(EXCLUDED.chunk_count, documents.chunk_count),
              updated_at = now()
            RETURNING *
            """,
            {
                "id": doc_pk,
                "collection_name": collection_name,
                "owner_id": owner_id,
                "doc_id": doc_id,
                "title": title,
                "filename": filename,
                "year": year,
                "pdf_object_key": pdf_object_key,
                "artifact_prefix": artifact_prefix,
                "source_document_id": source_document_id,
                "status": status,
                "task_id": task_id,
                "chunk_count": chunk_count,
            },
        )
        return Document.model_validate(cur.fetchone())


def get_document(collection_name: str, doc_id: str) -> Document | None:
    with base.cursor() as cur:
        cur.execute(
            "SELECT * FROM documents WHERE collection_name = %s AND doc_id = %s",
            (collection_name, doc_id),
        )
        return _model(Document, cur.fetchone())


def find_document_by_doc_id(doc_id: str, auth: AuthContext) -> Document | None:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT d.* FROM documents d
            JOIN kb_collections c ON c.name = d.collection_name
            WHERE d.doc_id = %(doc_id)s
              AND (
                c.owner_id = %(user_id)s
                OR c.visibility = 'public'
                OR (c.visibility = 'org' AND c.org_id IS NOT NULL AND c.org_id = %(org_id)s)
              )
            ORDER BY c.owner_id = %(user_id)s DESC, d.updated_at DESC
            LIMIT 1
            """,
            {"doc_id": doc_id, "user_id": auth.user_id, "org_id": auth.org_id},
        )
        return _model(Document, cur.fetchone())


def list_documents(collection_name: str, auth: AuthContext) -> list[Document]:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT d.* FROM documents d
            JOIN kb_collections c ON c.name = d.collection_name
            WHERE d.collection_name = %(collection_name)s
              AND (
                c.owner_id = %(user_id)s
                OR c.visibility = 'public'
                OR (c.visibility = 'org' AND c.org_id IS NOT NULL AND c.org_id = %(org_id)s)
              )
            ORDER BY d.updated_at DESC
            """,
            {"collection_name": collection_name, "user_id": auth.user_id, "org_id": auth.org_id},
        )
        return [Document.model_validate(r) for r in cur.fetchall()]


def delete_document(collection_name: str, doc_id: str, auth: AuthContext) -> bool:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM documents d
            USING kb_collections c
            WHERE d.collection_name = c.name
              AND d.collection_name = %(collection_name)s
              AND d.doc_id = %(doc_id)s
              AND c.owner_id = %(owner_id)s
            """,
            {"collection_name": collection_name, "doc_id": doc_id, "owner_id": auth.user_id},
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def upsert_skill_metadata(
    *,
    auth: AuthContext,
    skill_id: str,
    name: str,
    description: str | None = None,
    visibility: Visibility = "private",
    source_owner_id: str | None = None,
    source_skill_id: str | None = None,
) -> UserSkill:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_skills(
              owner_id, id, org_id, visibility, name, description,
              source_owner_id, source_skill_id, updated_at
            )
            VALUES (
              %(owner_id)s, %(id)s, %(org_id)s, %(visibility)s, %(name)s, %(description)s,
              %(source_owner_id)s, %(source_skill_id)s, now()
            )
            ON CONFLICT (owner_id, id) DO UPDATE SET
              org_id = EXCLUDED.org_id,
              name = EXCLUDED.name,
              description = EXCLUDED.description,
              updated_at = now()
            RETURNING *
            """,
            {
                "owner_id": auth.user_id,
                "id": skill_id,
                "org_id": auth.org_id,
                "visibility": visibility,
                "name": name,
                "description": description,
                "source_owner_id": source_owner_id,
                "source_skill_id": source_skill_id,
            },
        )
        return UserSkill.model_validate(cur.fetchone())


def list_skill_metadata(auth: AuthContext) -> list[UserSkill]:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM user_skills
            WHERE owner_id = %(user_id)s
               OR visibility = 'public'
               OR (visibility = 'org' AND org_id IS NOT NULL AND org_id = %(org_id)s)
            ORDER BY
              CASE
                WHEN visibility = 'public' THEN 0
                WHEN visibility = 'org' THEN 1
                WHEN owner_id = %(user_id)s THEN 2
                ELSE 3
              END,
              updated_at DESC
            """,
            {"user_id": auth.user_id, "org_id": auth.org_id},
        )
        return [UserSkill.model_validate(r) for r in cur.fetchall()]


def get_skill_metadata(owner_id: str, skill_id: str) -> UserSkill | None:
    with base.cursor() as cur:
        cur.execute("SELECT * FROM user_skills WHERE owner_id = %s AND id = %s", (owner_id, skill_id))
        return _model(UserSkill, cur.fetchone())


def find_readable_skill(skill_id: str, auth: AuthContext) -> UserSkill | None:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM user_skills
            WHERE id = %(id)s
              AND (
                owner_id = %(user_id)s
                OR visibility = 'public'
                OR (visibility = 'org' AND org_id IS NOT NULL AND org_id = %(org_id)s)
              )
            ORDER BY owner_id = %(user_id)s DESC, updated_at DESC
            LIMIT 1
            """,
            {"id": skill_id, "user_id": auth.user_id, "org_id": auth.org_id},
        )
        return _model(UserSkill, cur.fetchone())


def update_skill_visibility(skill_id: str, visibility: Visibility, auth: AuthContext) -> UserSkill | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE user_skills
            SET visibility = %(visibility)s, updated_at = now()
            WHERE owner_id = %(owner_id)s AND id = %(id)s
            RETURNING *
            """,
            {"owner_id": auth.user_id, "id": skill_id, "visibility": visibility},
        )
        return _model(UserSkill, cur.fetchone())


def delete_skill_metadata(skill_id: str, auth: AuthContext) -> bool:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM user_skills WHERE owner_id = %s AND id = %s",
            (auth.user_id, skill_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Conversations / messages / shares
# ---------------------------------------------------------------------------


def upsert_conversation(
    *,
    conversation_id: str,
    auth: AuthContext,
    title: str = "",
    visibility: Visibility = "private",
    active_leaf_message_id: str | None = None,
    session_id: str | None = None,
    forked_from: str | None = None,
) -> Conversation:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations(
              id, owner_id, org_id, visibility, title, active_leaf_message_id,
              session_id, forked_from, updated_at
            )
            VALUES (
              %(id)s, %(owner_id)s, %(org_id)s, %(visibility)s, %(title)s,
              %(active_leaf_message_id)s, %(session_id)s, %(forked_from)s, now()
            )
            ON CONFLICT (id) DO UPDATE SET
              title = COALESCE(NULLIF(EXCLUDED.title, ''), conversations.title),
              active_leaf_message_id = COALESCE(EXCLUDED.active_leaf_message_id, conversations.active_leaf_message_id),
              session_id = COALESCE(EXCLUDED.session_id, conversations.session_id),
              updated_at = now()
            RETURNING *
            """,
            {
                "id": conversation_id,
                "owner_id": auth.user_id,
                "org_id": auth.org_id,
                "visibility": visibility,
                "title": title,
                "active_leaf_message_id": active_leaf_message_id,
                "session_id": session_id,
                "forked_from": forked_from,
            },
        )
        return Conversation.model_validate(cur.fetchone())


def get_conversation(conversation_id: str) -> Conversation | None:
    with base.cursor() as cur:
        cur.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
        return _model(Conversation, cur.fetchone())


def set_conversation_active_leaf(
    conversation_id: str,
    active_leaf_message_id: str | None,
    *,
    session_id: str | None = None,
) -> Conversation | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversations
            SET active_leaf_message_id = %(active_leaf_message_id)s,
                session_id = COALESCE(%(session_id)s, session_id),
                updated_at = now()
            WHERE id = %(conversation_id)s
            RETURNING *
            """,
            {
                "conversation_id": conversation_id,
                "active_leaf_message_id": active_leaf_message_id,
                "session_id": session_id,
            },
        )
        return _model(Conversation, cur.fetchone())


def list_conversations(auth: AuthContext) -> list[Conversation]:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM conversations
            WHERE owner_id = %(user_id)s
               OR visibility = 'public'
               OR (visibility = 'org' AND org_id IS NOT NULL AND org_id = %(org_id)s)
            ORDER BY updated_at DESC
            """,
            {"user_id": auth.user_id, "org_id": auth.org_id},
        )
        return [Conversation.model_validate(r) for r in cur.fetchall()]


def update_conversation_visibility(
    conversation_id: str,
    visibility: Visibility,
    auth: AuthContext,
) -> Conversation | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversations
            SET visibility = %(visibility)s, updated_at = now()
            WHERE id = %(id)s AND owner_id = %(owner_id)s
            RETURNING *
            """,
            {"id": conversation_id, "owner_id": auth.user_id, "visibility": visibility},
        )
        return _model(Conversation, cur.fetchone())


def upsert_message(message: Message) -> Message:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages(
              id, conversation_id, parent_id, role, content, hits, context,
              research, usage, latency_s, params, status, error
            )
            VALUES (
              %(id)s, %(conversation_id)s, %(parent_id)s, %(role)s, %(content)s,
              %(hits)s, %(context)s, %(research)s, %(usage)s, %(latency_s)s,
              %(params)s, %(status)s, %(error)s
            )
            ON CONFLICT (id) DO UPDATE SET
              content = EXCLUDED.content,
              hits = EXCLUDED.hits,
              context = EXCLUDED.context,
              research = EXCLUDED.research,
              usage = EXCLUDED.usage,
              latency_s = EXCLUDED.latency_s,
              params = EXCLUDED.params,
              status = EXCLUDED.status,
              error = EXCLUDED.error
            RETURNING *
            """,
            {
                **message.model_dump(exclude={"created_at"}),
                "hits": _json(message.hits),
                "research": _json(message.research),
                "usage": _json(message.usage),
                "params": _json(message.params),
            },
        )
        return Message.model_validate(cur.fetchone())


def get_message(message_id: str) -> Message | None:
    with base.cursor() as cur:
        cur.execute("SELECT * FROM messages WHERE id = %s", (message_id,))
        return _model(Message, cur.fetchone())


def update_assistant_message(
    message_id: str,
    *,
    content: str | None = None,
    hits: list[dict[str, Any]] | None = None,
    context: str | None = None,
    research: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    latency_s: float | None = None,
    status: str | None = None,
    error: str | None = None,
) -> Message | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE messages
            SET content = COALESCE(%(content)s, content),
                hits = COALESCE(%(hits)s, hits),
                context = COALESCE(%(context)s, context),
                research = COALESCE(%(research)s, research),
                usage = COALESCE(%(usage)s, usage),
                latency_s = COALESCE(%(latency_s)s, latency_s),
                status = COALESCE(%(status)s, status),
                error = %(error)s
            WHERE id = %(id)s
            RETURNING *
            """,
            {
                "id": message_id,
                "content": content,
                "hits": _json(hits),
                "context": context,
                "research": _json(research),
                "usage": _json(usage),
                "latency_s": latency_s,
                "status": status,
                "error": error,
            },
        )
        return _model(Message, cur.fetchone())


def list_messages(conversation_id: str) -> list[Message]:
    with base.cursor() as cur:
        cur.execute(
            "SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
            (conversation_id,),
        )
        return [Message.model_validate(r) for r in cur.fetchall()]


def get_mainline_messages(conversation_id: str) -> list[Message]:
    conv = get_conversation(conversation_id)
    if not conv or not conv.active_leaf_message_id:
        return []
    with base.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE chain AS (
              SELECT *, 0 AS depth FROM messages WHERE id = %(leaf_id)s
              UNION ALL
              SELECT m.*, chain.depth + 1 AS depth
              FROM messages m
              JOIN chain ON chain.parent_id = m.id
            )
            SELECT id, conversation_id, parent_id, role, content, hits, context, research,
                   usage, latency_s, params, status, error, created_at
            FROM chain
            ORDER BY depth DESC
            """,
            {"leaf_id": conv.active_leaf_message_id},
        )
        return [Message.model_validate(r) for r in cur.fetchall()]


def get_message_chain_to(message_id: str | None) -> list[Message]:
    if not message_id:
        return []
    with base.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE chain AS (
              SELECT *, 0 AS depth FROM messages WHERE id = %(message_id)s
              UNION ALL
              SELECT m.*, chain.depth + 1 AS depth
              FROM messages m
              JOIN chain ON chain.parent_id = m.id
            )
            SELECT id, conversation_id, parent_id, role, content, hits, context, research,
                   usage, latency_s, params, status, error, created_at
            FROM chain
            ORDER BY depth DESC
            """,
            {"message_id": message_id},
        )
        return [Message.model_validate(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Generation runs / message events
# ---------------------------------------------------------------------------


def create_generation_run(
    *,
    run_id: str,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    owner_id: str,
    org_id: str | None = None,
    params: dict[str, Any] | None = None,
    redis_stream: str | None = None,
    artifact_prefix: str | None = None,
) -> GenerationRun:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO generation_runs(
              id, conversation_id, user_message_id, assistant_message_id, owner_id,
              org_id, status, params, redis_stream, artifact_prefix, updated_at
            )
            VALUES (
              %(id)s, %(conversation_id)s, %(user_message_id)s, %(assistant_message_id)s,
              %(owner_id)s, %(org_id)s, 'queued', %(params)s, %(redis_stream)s,
              %(artifact_prefix)s, now()
            )
            RETURNING *
            """,
            {
                "id": run_id,
                "conversation_id": conversation_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "owner_id": owner_id,
                "org_id": org_id,
                "params": _json(params),
                "redis_stream": redis_stream,
                "artifact_prefix": artifact_prefix,
            },
        )
        return GenerationRun.model_validate(cur.fetchone())


def get_generation_run(run_id: str) -> GenerationRun | None:
    with base.cursor() as cur:
        cur.execute("SELECT * FROM generation_runs WHERE id = %s", (run_id,))
        return _model(GenerationRun, cur.fetchone())


def mark_generation_run_running(run_id: str) -> GenerationRun | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE generation_runs
            SET status = 'running',
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE id = %(id)s AND status = 'queued' AND cancel_requested IS FALSE
            RETURNING *
            """,
            {"id": run_id},
        )
        return _model(GenerationRun, cur.fetchone())


def update_generation_run_status(
    run_id: str,
    status: RunStatus,
    *,
    error: str | None = None,
) -> GenerationRun | None:
    terminal = status in {"done", "failed", "stopped"}
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE generation_runs
            SET status = %(status)s,
                error = %(error)s,
                updated_at = now(),
                started_at = CASE
                  WHEN %(status)s = 'running' THEN COALESCE(started_at, now())
                  ELSE started_at
                END,
                finished_at = CASE
                  WHEN %(terminal)s THEN COALESCE(finished_at, now())
                  ELSE finished_at
                END
            WHERE id = %(id)s
            RETURNING *
            """,
            {"id": run_id, "status": status, "error": error, "terminal": terminal},
        )
        return _model(GenerationRun, cur.fetchone())


def request_generation_run_stop(run_id: str) -> GenerationRun | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE generation_runs
            SET cancel_requested = true,
                status = CASE WHEN status = 'queued' THEN 'stopped' ELSE status END,
                finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END,
                updated_at = now()
            WHERE id = %(id)s
            RETURNING *
            """,
            {"id": run_id},
        )
        return _model(GenerationRun, cur.fetchone())


def generation_run_should_stop(run_id: str) -> bool:
    with base.cursor() as cur:
        cur.execute(
            "SELECT cancel_requested FROM generation_runs WHERE id = %s",
            (run_id,),
        )
        row = cur.fetchone()
        return bool(row and row.get("cancel_requested"))


def append_message_event(run_id: str, event_type: str, payload: dict[str, Any]) -> MessageEvent:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH next_seq AS (
              SELECT COALESCE(MAX(seq), 0) + 1 AS seq
              FROM message_events
              WHERE run_id = %(run_id)s
            )
            INSERT INTO message_events(run_id, seq, type, payload)
            SELECT %(run_id)s, next_seq.seq, %(type)s, %(payload)s
            FROM next_seq
            RETURNING *
            """,
            {"run_id": run_id, "type": event_type, "payload": _json(payload)},
        )
        return MessageEvent.model_validate(cur.fetchone())


def list_message_events(run_id: str, *, after_seq: int = 0, limit: int = 1000) -> list[MessageEvent]:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM message_events
            WHERE run_id = %(run_id)s AND seq > %(after_seq)s
            ORDER BY seq ASC
            LIMIT %(limit)s
            """,
            {"run_id": run_id, "after_seq": after_seq, "limit": limit},
        )
        return [MessageEvent.model_validate(r) for r in cur.fetchall()]


def copy_conversation_mainline_to_owner(source_id: str, auth: AuthContext) -> Conversation:
    source = get_conversation(source_id)
    if source is None:
        raise ValueError("conversation not found")
    source_messages = get_mainline_messages(source_id)
    new_id = f"c_{uuid.uuid4().hex[:16]}"
    id_map: dict[str, str] = {}
    new_active: str | None = None
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations(id, owner_id, org_id, visibility, title, forked_from, updated_at)
            VALUES (%s, %s, %s, 'private', %s, %s, now())
            RETURNING *
            """,
            (new_id, auth.user_id, auth.org_id, source.title, source.id),
        )
        conv = Conversation.model_validate(cur.fetchone())
        for msg in source_messages:
            new_msg_id = f"m_{uuid.uuid4().hex[:16]}"
            id_map[msg.id] = new_msg_id
            new_parent = id_map.get(msg.parent_id or "")
            cur.execute(
                """
                INSERT INTO messages(
                  id, conversation_id, parent_id, role, content, hits, context,
                  research, usage, latency_s, params, status, error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    new_msg_id,
                    new_id,
                    new_parent,
                    msg.role,
                    msg.content,
                    _json(msg.hits),
                    msg.context,
                    _json(msg.research),
                    _json(msg.usage),
                    msg.latency_s,
                    _json(msg.params),
                    msg.status,
                    msg.error,
                ),
            )
            new_active = new_msg_id
        cur.execute(
            """
            UPDATE conversations
            SET active_leaf_message_id = %s, updated_at = now()
            WHERE id = %s
            RETURNING *
            """,
            (new_active, new_id),
        )
        return Conversation.model_validate(cur.fetchone() or conv.model_dump())


def create_share(conversation_id: str, auth: AuthContext) -> ConversationShare | None:
    token = secrets.token_urlsafe(24)
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversation_shares(token, conversation_id, owner_id)
            SELECT %(token)s, id, owner_id FROM conversations
            WHERE id = %(conversation_id)s AND owner_id = %(owner_id)s
            RETURNING *
            """,
            {"token": token, "conversation_id": conversation_id, "owner_id": auth.user_id},
        )
        row = cur.fetchone()
        if not row:
            return None
        return ConversationShare.model_validate(row)


def revoke_share(conversation_id: str, auth: AuthContext) -> bool:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversation_shares
            SET revoked_at = now()
            WHERE conversation_id = %(conversation_id)s
              AND owner_id = %(owner_id)s
              AND revoked_at IS NULL
            """,
            {"conversation_id": conversation_id, "owner_id": auth.user_id},
        )
        return cur.rowcount > 0


def get_share(token: str) -> ConversationShare | None:
    with base.cursor() as cur:
        cur.execute(
            "SELECT * FROM conversation_shares WHERE token = %s AND revoked_at IS NULL",
            (token,),
        )
        return _model(ConversationShare, cur.fetchone())


# ---------------------------------------------------------------------------
# Ingest tasks / items / events
# ---------------------------------------------------------------------------


def create_ingest_task(
    *,
    task_id: str,
    auth: AuthContext,
    collection_name: str,
    kind: str = "upload",
    total_items: int = 0,
    params: dict[str, Any] | None = None,
    redis_stream: str | None = None,
) -> IngestTask:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_tasks(
              id, owner_id, org_id, collection_name, kind, status,
              total_items, params, redis_stream, updated_at
            )
            VALUES (
              %(id)s, %(owner_id)s, %(org_id)s, %(collection_name)s, %(kind)s,
              'queued', %(total_items)s, %(params)s, %(redis_stream)s, now()
            )
            RETURNING *
            """,
            {
                "id": task_id,
                "owner_id": auth.user_id,
                "org_id": auth.org_id,
                "collection_name": collection_name,
                "kind": kind,
                "total_items": total_items,
                "params": _json(params),
                "redis_stream": redis_stream,
            },
        )
        return IngestTask.model_validate(cur.fetchone())


def add_ingest_task_item(
    *,
    item_id: str,
    task_id: str,
    collection_name: str,
    owner_id: str,
    doc_id: str,
    filename: str | None = None,
    pdf_path: str | None = None,
    doc_dir: str | None = None,
    pdf_object_key: str | None = None,
    artifact_prefix: str | None = None,
    status: IngestItemStatus = "pending",
    error: str | None = None,
) -> IngestTaskItem:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_task_items(
              id, task_id, collection_name, owner_id, doc_id, filename,
              pdf_path, doc_dir, pdf_object_key, artifact_prefix, status, error, updated_at
            )
            VALUES (
              %(id)s, %(task_id)s, %(collection_name)s, %(owner_id)s, %(doc_id)s,
              %(filename)s, %(pdf_path)s, %(doc_dir)s, %(pdf_object_key)s,
              %(artifact_prefix)s, %(status)s, %(error)s, now()
            )
            RETURNING *
            """,
            {
                "id": item_id,
                "task_id": task_id,
                "collection_name": collection_name,
                "owner_id": owner_id,
                "doc_id": doc_id,
                "filename": filename,
                "pdf_path": pdf_path,
                "doc_dir": doc_dir,
                "pdf_object_key": pdf_object_key,
                "artifact_prefix": artifact_prefix,
                "status": status,
                "error": error,
            },
        )
        return IngestTaskItem.model_validate(cur.fetchone())


def get_ingest_task(task_id: str) -> IngestTask | None:
    with base.cursor() as cur:
        cur.execute("SELECT * FROM ingest_tasks WHERE id = %s", (task_id,))
        return _model(IngestTask, cur.fetchone())


def list_ingest_task_items(task_id: str) -> list[IngestTaskItem]:
    with base.cursor() as cur:
        cur.execute(
            "SELECT * FROM ingest_task_items WHERE task_id = %s ORDER BY created_at ASC",
            (task_id,),
        )
        return [IngestTaskItem.model_validate(r) for r in cur.fetchall()]


def mark_ingest_task_running(task_id: str) -> IngestTask | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_tasks
            SET status = 'running',
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE id = %(id)s AND status = 'queued' AND cancel_requested IS FALSE
            RETURNING *
            """,
            {"id": task_id},
        )
        return _model(IngestTask, cur.fetchone())


def update_ingest_task_counts(task_id: str) -> IngestTask | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH counts AS (
              SELECT
                count(*) AS total,
                count(*) FILTER (WHERE status IN ('ready', 'failed', 'cancelled', 'skipped')) AS completed,
                count(*) FILTER (WHERE status = 'failed') AS failed,
                count(*) FILTER (WHERE status = 'skipped') AS skipped
              FROM ingest_task_items
              WHERE task_id = %(id)s
            )
            UPDATE ingest_tasks
            SET total_items = counts.total,
                completed_items = counts.completed,
                failed_items = counts.failed,
                skipped_items = counts.skipped,
                progress = CASE WHEN counts.total > 0
                  THEN counts.completed::double precision / counts.total::double precision
                  ELSE 1 END,
                updated_at = now()
            FROM counts
            WHERE ingest_tasks.id = %(id)s
            RETURNING ingest_tasks.*
            """,
            {"id": task_id},
        )
        return _model(IngestTask, cur.fetchone())


def update_ingest_task_status(
    task_id: str,
    status: IngestTaskStatus,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> IngestTask | None:
    terminal = status in {"done", "failed", "cancelled"}
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_tasks
            SET status = %(status)s,
                result = COALESCE(%(result)s, result),
                error = %(error)s,
                updated_at = now(),
                started_at = CASE
                  WHEN %(status)s = 'running' THEN COALESCE(started_at, now())
                  ELSE started_at
                END,
                finished_at = CASE
                  WHEN %(terminal)s THEN COALESCE(finished_at, now())
                  ELSE finished_at
                END
            WHERE id = %(id)s
            RETURNING *
            """,
            {
                "id": task_id,
                "status": status,
                "result": _json(result),
                "error": error,
                "terminal": terminal,
            },
        )
        return _model(IngestTask, cur.fetchone())


def request_ingest_task_cancel(task_id: str) -> IngestTask | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_tasks
            SET cancel_requested = true,
                status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END,
                updated_at = now()
            WHERE id = %(id)s
            RETURNING *
            """,
            {"id": task_id},
        )
        return _model(IngestTask, cur.fetchone())


def ingest_task_should_cancel(task_id: str) -> bool:
    with base.cursor() as cur:
        cur.execute("SELECT cancel_requested FROM ingest_tasks WHERE id = %s", (task_id,))
        row = cur.fetchone()
        return bool(row and row.get("cancel_requested"))


def update_ingest_task_item(
    item_id: str,
    *,
    status: IngestItemStatus | None = None,
    error: str | None = None,
    chunk_count: int | None = None,
    artifact_prefix: str | None = None,
) -> IngestTaskItem | None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_task_items
            SET status = COALESCE(%(status)s, status),
                error = %(error)s,
                chunk_count = COALESCE(%(chunk_count)s, chunk_count),
                artifact_prefix = COALESCE(%(artifact_prefix)s, artifact_prefix),
                started_at = CASE
                  WHEN %(status)s = 'running' THEN COALESCE(started_at, now())
                  ELSE started_at
                END,
                finished_at = CASE
                  WHEN %(status)s IN ('ready', 'failed', 'cancelled', 'skipped')
                  THEN COALESCE(finished_at, now())
                  ELSE finished_at
                END,
                updated_at = now()
            WHERE id = %(id)s
            RETURNING *
            """,
            {
                "id": item_id,
                "status": status,
                "error": error,
                "chunk_count": chunk_count,
                "artifact_prefix": artifact_prefix,
            },
        )
        return _model(IngestTaskItem, cur.fetchone())


def cancel_pending_ingest_items(task_id: str) -> None:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_task_items
            SET status = 'cancelled', updated_at = now(), finished_at = now()
            WHERE task_id = %s AND status = 'pending'
            """,
            (task_id,),
        )


def append_ingest_task_event(
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> IngestTaskEvent:
    with base.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH next_seq AS (
              SELECT COALESCE(MAX(seq), 0) + 1 AS seq
              FROM ingest_task_events
              WHERE task_id = %(task_id)s
            )
            INSERT INTO ingest_task_events(task_id, seq, type, payload)
            SELECT %(task_id)s, next_seq.seq, %(type)s, %(payload)s
            FROM next_seq
            RETURNING *
            """,
            {"task_id": task_id, "type": event_type, "payload": _json(payload)},
        )
        return IngestTaskEvent.model_validate(cur.fetchone())


def list_ingest_task_events(
    task_id: str,
    *,
    after_seq: int = 0,
    limit: int = 1000,
) -> list[IngestTaskEvent]:
    with base.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM ingest_task_events
            WHERE task_id = %(task_id)s AND seq > %(after_seq)s
            ORDER BY seq ASC
            LIMIT %(limit)s
            """,
            {"task_id": task_id, "after_seq": after_seq, "limit": limit},
        )
        return [IngestTaskEvent.model_validate(r) for r in cur.fetchall()]
