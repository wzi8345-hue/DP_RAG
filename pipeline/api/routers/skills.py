"""专家技能 (Skill) 管理: 列表 / 模版 / 新建编辑 / 删除。

文件定义式 skill 存放于配置 professional.skills.dirs (内置) 与 upload_dir (用户上传)。
本路由让用户通过 UI 维护 upload_dir 下的自定义 skill; 内置 skill 只读 (editable=false)。
保存/删除后清空研究 agent 缓存, 下次专业模式请求即重载。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_pipeline, verify_api_key
from ..models import (
    SkillDeleteResponse,
    SkillListResponse,
    SkillSaveResponse,
    SkillSpec,
    SkillSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _skills_cfg(pipe):
    from ...routing.research_skills import resolve_skills_config
    prof = (
        (pipe.config.retrieval.get("langgraph", {}) or {}).get("professional", {}) or {}
    )
    return resolve_skills_config(prof.get("skills", {}) or {})


@router.get("/skills", response_model=SkillListResponse)
def list_skills(_auth: str = Depends(verify_api_key)) -> SkillListResponse:
    """列出所有已加载的 skill (内置 + 用户), 含可编辑提示词正文。"""
    from ...routing.research_skills import load_skills, skill_to_summary

    pipe = get_pipeline()
    cfg = _skills_cfg(pipe)
    skills = load_skills(cfg["dirs"]) if cfg["enabled"] else {}
    summaries = [
        SkillSummary(**skill_to_summary(s, upload_dir=cfg["upload_dir"]))
        for s in sorted(skills.values(), key=lambda s: (-s.priority, s.id))
    ]
    return SkillListResponse(
        enabled=cfg["enabled"],
        router_mode=cfg["router_mode"],
        upload_dir=cfg["upload_dir"],
        skills=summaries,
    )


@router.get("/skills/template")
def get_skill_template(_auth: str = Depends(verify_api_key)) -> dict:
    """返回新建 skill 的填写模版 (字段说明 + 示例)。"""
    from ...routing.research_skills import skill_template

    return skill_template()


@router.post("/skills", response_model=SkillSaveResponse)
def save_skill(spec: SkillSpec, _auth: str = Depends(verify_api_key)) -> SkillSaveResponse:
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
    return SkillSaveResponse(saved=True, id=saved.id, skill=summary)


@router.delete("/skills/{skill_id}", response_model=SkillDeleteResponse)
def remove_skill(skill_id: str, _auth: str = Depends(verify_api_key)) -> SkillDeleteResponse:
    """删除一个用户 skill (仅限 upload_dir; 内置 skill 不可删)。"""
    from ...routing.research_skills import delete_skill

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
    return SkillDeleteResponse(deleted=True, id=skill_id)
