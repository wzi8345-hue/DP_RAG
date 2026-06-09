"""专家技能 (Skill) 管理: 列表 / 模版 / 新建编辑 / 删除。

文件定义式 skill 存放于配置 professional.skills.dirs (内置) 与 upload_dir (用户上传)。
本路由让用户通过 UI 维护 upload_dir 下的自定义 skill; 内置 skill 只读 (editable=false)。
保存/删除后清空研究 agent 缓存, 下次专业模式请求即重载。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..deps import AuthContext, get_pipeline, require_auth
from ..models import (
    ResourceCopyRequest,
    ResourceCopyResponse,
    SkillDeleteResponse,
    SkillListResponse,
    SkillSaveResponse,
    SkillSpec,
    SkillSummary,
    VisibilityRequest,
)
from ...db import repo

logger = logging.getLogger(__name__)

router = APIRouter()


def _skills_cfg(pipe):
    from ...routing.research_skills import resolve_skills_config
    prof = (
        (pipe.config.retrieval.get("langgraph", {}) or {}).get("professional", {}) or {}
    )
    return resolve_skills_config(prof.get("skills", {}) or {})


@router.get("/skills", response_model=SkillListResponse)
def list_skills(auth: AuthContext = Depends(require_auth)) -> SkillListResponse:
    """列出所有已加载的 skill (内置 + 用户), 含可编辑提示词正文。"""
    from ...routing.research_skills import load_skills, skill_to_summary

    pipe = get_pipeline()
    cfg = _skills_cfg(pipe)
    skills = load_skills(cfg["dirs"]) if cfg["enabled"] else {}
    meta_by_id = {}
    if repo.available():
        meta_by_id = {s.id: s for s in repo.list_skill_metadata(auth)}
    summaries = []
    for s in sorted(skills.values(), key=lambda s: (-s.priority, s.id)):
        raw = skill_to_summary(s, upload_dir=cfg["upload_dir"])
        editable = bool(raw.get("editable"))
        meta = meta_by_id.get(s.id)
        if editable and repo.available() and meta is None:
            # legacy uploaded skill: first visible owner claims it.
            meta = repo.upsert_skill_metadata(
                auth=auth,
                skill_id=s.id,
                name=s.name,
                description=s.description,
            )
        if editable and repo.available() and meta is None:
            continue
        if meta:
            raw.update(
                {
                    "owner_id": meta.owner_id,
                    "org_id": meta.org_id,
                    "visibility": meta.visibility,
                    "mine": meta.owner_id == auth.user_id,
                    "editable": meta.owner_id == auth.user_id,
                }
            )
        elif not editable:
            raw.update({"visibility": "public", "mine": False})
        summaries.append(SkillSummary(**raw))
    return SkillListResponse(
        enabled=cfg["enabled"],
        router_mode=cfg["router_mode"],
        upload_dir=cfg["upload_dir"],
        skills=summaries,
    )


@router.get("/skills/template")
def get_skill_template(_auth: str = Depends(require_auth)) -> dict:
    """返回新建 skill 的填写模版 (字段说明 + 示例)。"""
    from ...routing.research_skills import skill_template

    return skill_template()


@router.post("/skills", response_model=SkillSaveResponse)
def save_skill(spec: SkillSpec, auth: AuthContext = Depends(require_auth)) -> SkillSaveResponse:
    """新建或覆盖一个用户 skill (写入 upload_dir), 并触发研究 agent 重载。"""
    from ...routing.research_skills import (
        load_skills,
        parse_skill_dir,
        skill_to_summary,
        write_skill,
    )

    pipe = get_pipeline()
    cfg = _skills_cfg(pipe)
    if not cfg["enabled"]:
        raise HTTPException(status_code=400, detail="skills 功能未启用 (professional.skills.enabled=false)")

    try:
        skill_dir = write_skill(cfg["upload_dir"], spec.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover
        logger.exception("[skills] 写入失败")
        raise HTTPException(status_code=500, detail=f"写入失败: {e}")

    # 触发重载, 让本次保存即时生效
    try:
        pipe._get_query_flow().reload_skills()
    except Exception as e:  # pragma: no cover
        logger.warning(f"[skills] reload_skills 失败 (不影响写入): {e}")

    saved = parse_skill_dir(skill_dir)
    if saved is None:
        raise HTTPException(status_code=500, detail="写入后解析失败, 请检查内容")
    # 重新加载以判定 editable (源目录归属)
    _ = load_skills(cfg["dirs"])
    summary = SkillSummary(**skill_to_summary(saved, upload_dir=cfg["upload_dir"]))
    if repo.available():
        meta = repo.upsert_skill_metadata(
            auth=auth,
            skill_id=saved.id,
            name=saved.name,
            description=saved.description,
        )
        summary.owner_id = meta.owner_id
        summary.org_id = meta.org_id
        summary.visibility = meta.visibility
        summary.mine = True
        summary.editable = True
    return SkillSaveResponse(saved=True, id=saved.id, skill=summary)


@router.delete("/skills/{skill_id}", response_model=SkillDeleteResponse)
def remove_skill(skill_id: str, _auth: AuthContext = Depends(require_auth)) -> SkillDeleteResponse:
    """删除一个用户 skill (仅限 upload_dir; 内置 skill 不可删)。"""
    from ...routing.research_skills import delete_skill

    auth = _auth
    pipe = get_pipeline()
    cfg = _skills_cfg(pipe)
    try:
        deleted = delete_skill(cfg["upload_dir"], skill_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="未找到可删除的用户 skill (内置 skill 不可删除)",
        )
    try:
        pipe._get_query_flow().reload_skills()
    except Exception as e:  # pragma: no cover
        logger.warning(f"[skills] reload_skills 失败: {e}")
    if repo.available():
        repo.delete_skill_metadata(skill_id, auth)
    return SkillDeleteResponse(deleted=True, id=skill_id)


@router.patch("/skills/{skill_id}/visibility")
def set_skill_visibility(
    skill_id: str,
    req: VisibilityRequest,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    updated = repo.update_skill_visibility(skill_id, req.visibility, auth)
    if updated is None:
        raise HTTPException(status_code=404, detail="未找到可写的 skill")
    return {"updated": True, "id": skill_id, "visibility": updated.visibility}


@router.post("/skills/copy-to-mine", response_model=ResourceCopyResponse)
def copy_skill_to_mine(
    req: ResourceCopyRequest,
    auth: AuthContext = Depends(require_auth),
) -> ResourceCopyResponse:
    from ...routing.research_skills import load_skills, skill_to_summary, write_skill

    pipe = get_pipeline()
    cfg = _skills_cfg(pipe)
    if not cfg["enabled"]:
        raise HTTPException(status_code=400, detail="skills 功能未启用")
    if repo.available():
        source_meta = repo.find_readable_skill(req.id, auth)
        if source_meta and source_meta.owner_id == auth.user_id:
            return ResourceCopyResponse(id=source_meta.id, name=source_meta.name)

    skills = load_skills(cfg["dirs"])
    source = skills.get(req.id)
    if source is None:
        raise HTTPException(status_code=404, detail="skill 不存在或不可读")
    new_id = f"{source.id}_copy"
    i = 2
    existing = set(skills)
    while new_id in existing:
        new_id = f"{source.id}_copy_{i}"
        i += 1
    spec = skill_to_summary(source, upload_dir=cfg["upload_dir"])
    spec["id"] = new_id
    spec["name"] = f"{source.name} copy"
    write_skill(cfg["upload_dir"], spec)
    if repo.available():
        repo.upsert_skill_metadata(
            auth=auth,
            skill_id=new_id,
            name=spec["name"],
            description=spec.get("description"),
            source_owner_id=(source_meta.owner_id if "source_meta" in locals() and source_meta else None),
            source_skill_id=req.id,
        )
    try:
        pipe._get_query_flow().reload_skills()
    except Exception as e:  # pragma: no cover
        logger.warning("[skills] reload_skills 失败: %s", e)
    return ResourceCopyResponse(id=new_id, name=spec["name"])
