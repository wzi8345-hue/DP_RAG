"""Conversation CRUD, visibility, sharing and copy-on-continue helpers."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from ..authz import require_manage, require_read, require_visibility_allowed
from ..deps import AuthContext, require_auth
from ..models import (
    ConversationCopyRequest,
    ConversationIdRequest,
    ConversationShareRequest,
    ConversationShareResponse,
    ConversationVisibilityRequest,
    SharedConversationRequest,
)
from ...db import repo

router = APIRouter()


def _frontend_base_url() -> str:
    return os.environ.get("FRONTEND_BASE_URL", "http://localhost:9527").rstrip("/")


def _message_payload(m) -> dict:
    return {
        "id": m.id,
        "parentId": m.parent_id,
        "role": m.role,
        "content": m.content,
        "hits": m.hits or [],
        "context": m.context,
        "research": m.research,
        "latency": m.latency_s,
        "usage": m.usage,
        "status": m.status,
        "error": m.error,
        "createdAt": int(m.created_at.timestamp() * 1000) if m.created_at else 0,
    }


def _conversation_payload(conv, auth: AuthContext | None = None, *, include_messages: bool = False) -> dict:
    payload = {
        "id": conv.id,
        "title": conv.title,
        "sessionId": conv.session_id,
        "visibility": conv.visibility,
        "activeLeafId": conv.active_leaf_message_id,
        "updatedAt": int(conv.updated_at.timestamp() * 1000) if conv.updated_at else 0,
        "ownerId": conv.owner_id,
        "mine": bool(auth and conv.owner_id == auth.user_id),
        "forkedFrom": conv.forked_from,
    }
    if include_messages:
        messages = repo.list_messages(conv.id)
        payload["messages"] = {m.id: _message_payload(m) for m in messages}
        payload["rootIds"] = [m.id for m in messages if m.parent_id is None]
    return payload


@router.post("/conversations/list")
def list_conversations(auth: AuthContext = Depends(require_auth)) -> dict:
    if not repo.available():
        return {"conversations": []}
    return {"conversations": [_conversation_payload(c, auth) for c in repo.list_conversations(auth)]}


@router.post("/conversations/get")
def get_conversation(
    req: ConversationIdRequest,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    conversation_id = req.conversation_id
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    conv = repo.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    require_read(auth, conv)
    return {"conversation": _conversation_payload(conv, auth, include_messages=True)}


@router.post("/conversations/set-visibility")
def set_conversation_visibility(
    req: ConversationVisibilityRequest,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    conversation_id = req.conversation_id
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    require_visibility_allowed(auth, req.visibility)
    conv = repo.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="未找到可写的对话")
    require_manage(auth, conv)
    updated = repo.update_conversation_visibility_as(conversation_id, req.visibility)
    if updated is None:
        raise HTTPException(status_code=404, detail="未找到可写的对话")
    if conv.owner_id != auth.user_id:
        repo.append_audit_log(
            auth=auth,
            resource_type="conversation",
            resource_id=conversation_id,
            action="set_visibility",
            target_owner_id=conv.owner_id,
            metadata={"visibility": req.visibility},
        )
    return {"updated": True, "conversation_id": conversation_id, "visibility": updated.visibility}


@router.post("/conversations/share", response_model=ConversationShareResponse)
def share_conversation(
    req: ConversationShareRequest,
    auth: AuthContext = Depends(require_auth),
) -> ConversationShareResponse:
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    share = repo.create_share(req.conversation_id, auth)
    if share is None:
        raise HTTPException(status_code=404, detail="未找到可分享的对话")
    return ConversationShareResponse(token=share.token, url=f"{_frontend_base_url()}/s/{share.token}")


@router.post("/conversations/unshare")
def unshare_conversation(
    req: ConversationShareRequest,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    return {"revoked": repo.revoke_share(req.conversation_id, auth)}


@router.post("/conversations/shared/get")
def get_shared_conversation(req: SharedConversationRequest) -> dict:
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    share = repo.get_share(req.token)
    if share is None:
        raise HTTPException(status_code=404, detail="分享链接已失效")
    conv = repo.get_conversation(share.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"conversation": _conversation_payload(conv, None, include_messages=True)}


@router.post("/conversations/copy-to-mine")
def copy_conversation_to_mine(
    req: ConversationCopyRequest,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    source_id = req.conversation_id
    if req.token:
        share = repo.get_share(req.token)
        if share is None:
            raise HTTPException(status_code=404, detail="分享链接已失效")
        source_id = share.conversation_id
    if not source_id:
        raise HTTPException(status_code=400, detail="缺少 conversation_id 或 token")
    conv = repo.get_conversation(source_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    if conv.owner_id == auth.user_id:
        return {"conversation_id": conv.id}
    if not req.token:
        require_read(auth, conv)
    copied = repo.copy_conversation_mainline_to_owner(source_id, auth)
    return {"conversation_id": copied.id}
