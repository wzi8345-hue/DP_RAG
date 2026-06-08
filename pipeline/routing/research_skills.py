"""专家模式 (professional) 的文件定义式研究技能 (skill)。

每个 skill 是 pipeline/skills/<id>/ 下的一个文件夹:
  SKILL.md     —— frontmatter(元数据/触发词/充分性/调参/守卫) + Description(分类说明)
  plan.md      —— 规划提示词 (替换通用拆解段)
  policy.md    —— 策略提示词 (替换通用判断段)
  synthesis.md —— 可选, 综述输出结构 (## System / ## Thinking / ## User)

设计要点:
  - 本模块**不 import** routing.research, 只持有提示词正文字符串与元数据, 由 research_agent
    在节点里把它们组合进通用提示词框架, 因此对现有链路零耦合;
  - skill_router 选不到合适 skill 时返回 None, 上层回退现有通用 plan/policy/synthesis;
  - 加载/解析任何失败都降级 (跳过坏 skill / 回退通用), 不抛异常拖垮检索链路。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ResearchSkill:
    id: str
    name: str
    description: str = ""
    priority: int = 0
    triggers: List[str] = field(default_factory=list)
    # 提示词正文 (plan/policy 为"替换段"; synthesis 为整段 system/thinking/user)
    plan_system: str = ""
    policy_system: str = ""
    synthesis_system: str = ""
    synthesis_thinking_system: str = ""
    synthesis_user_template: str = ""
    # 收口标准 / 检索引导 / 调参 / 守卫
    default_sufficiency: Dict[str, Any] = field(default_factory=dict)
    prefer_first_paths: List[str] = field(default_factory=list)
    guards: List[str] = field(default_factory=list)
    max_rounds: Optional[int] = None
    max_batches: Optional[int] = None
    gap_stall_limit: Optional[int] = None
    stall_quality_floor: Optional[float] = None
    source_dir: str = ""


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _split_frontmatter(text: str) -> "Tuple[Dict[str, Any], str]":
    """拆出 YAML frontmatter 与正文; 无 frontmatter 时返回 ({}, 全文)。"""
    m = _FRONTMATTER_RE.match(text or "")
    if not m:
        return {}, (text or "")
    if yaml is None:
        return {}, m.group(2)
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception as e:  # pragma: no cover
        logger.warning(f"[skills] frontmatter 解析失败: {e}")
        meta = {}
    return meta, m.group(2)


def _split_sections(body: str) -> Dict[str, str]:
    """按 markdown 二级标题 (## X) 切分正文 → {小写标题: 内容}。"""
    sections: Dict[str, str] = {}
    if not body:
        return sections
    cur: Optional[str] = None
    buf: List[str] = []
    for line in body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = m.group(1).strip().lower()
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()
    return sections


def parse_skill_dir(d: str) -> Optional[ResearchSkill]:
    """解析单个 skill 文件夹; 缺 SKILL.md / plan.md / policy.md 则跳过 (返回 None)。"""
    skill_md = os.path.join(d, "SKILL.md")
    if not os.path.isfile(skill_md):
        return None
    meta, body = _split_frontmatter(_read(skill_md))
    sid = str(meta.get("id") or os.path.basename(d.rstrip("/"))).strip()
    if not sid:
        return None

    plan_body = _read(os.path.join(d, "plan.md")).strip()
    policy_body = _read(os.path.join(d, "policy.md")).strip()
    if not plan_body or not policy_body:
        logger.warning(f"[skills] 跳过 {sid}: 缺 plan.md 或 policy.md")
        return None

    sb_sections = _split_sections(_read(os.path.join(d, "synthesis.md")))
    desc_sections = _split_sections(body)
    description = desc_sections.get("description") or body.strip()

    tuning = meta.get("tuning") or {}
    if not isinstance(tuning, dict):
        tuning = {}

    def _opt_int(v: Any) -> Optional[int]:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _opt_float(v: Any) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return ResearchSkill(
        id=sid,
        name=str(meta.get("name") or sid),
        description=str(description or "").strip(),
        priority=_opt_int(meta.get("priority")) or 0,
        triggers=[str(t) for t in (meta.get("triggers") or []) if str(t).strip()],
        plan_system=plan_body,
        policy_system=policy_body,
        synthesis_system=sb_sections.get("system", ""),
        synthesis_thinking_system=sb_sections.get("thinking", ""),
        synthesis_user_template=sb_sections.get("user", ""),
        default_sufficiency=(
            meta.get("sufficiency") if isinstance(meta.get("sufficiency"), dict) else {}
        ),
        prefer_first_paths=[
            str(p) for p in (meta.get("prefer_first_paths") or []) if str(p).strip()
        ],
        guards=[str(g) for g in (meta.get("guards") or []) if str(g).strip()],
        max_rounds=_opt_int(tuning.get("max_rounds")),
        max_batches=_opt_int(tuning.get("max_batches")),
        gap_stall_limit=_opt_int(tuning.get("gap_stall_limit")),
        stall_quality_floor=_opt_float(tuning.get("stall_quality_floor")),
        source_dir=d,
    )


def load_skills(dirs: List[str]) -> Dict[str, ResearchSkill]:
    """扫描多个目录加载 skill; 同 id 后加载覆盖先加载 (用户目录可覆盖内置)。"""
    out: Dict[str, ResearchSkill] = {}
    for raw in dirs or []:
        d = os.path.expanduser(str(raw))
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.startswith(".") or name.startswith("__"):
                continue
            sub = os.path.join(d, name)
            if not os.path.isdir(sub):
                continue
            try:
                skill = parse_skill_dir(sub)
            except Exception as e:  # pragma: no cover
                logger.warning(f"[skills] 解析 {sub} 失败, 跳过: {e}")
                continue
            if skill:
                out[skill.id] = skill
    logger.info(f"[skills] 加载 {len(out)} 个 skill: {sorted(out)}")
    return out


# ---------------------------------------------------------------------------
# skill 选择 (skill_router): 思考型判定 + 强匹配门控
# ---------------------------------------------------------------------------

DEFAULT_MIN_CONFIDENCE = 0.6     # 思考模型给出的"强匹配"置信度下限
DEFAULT_STRONG_MIN_HITS = 2      # 触发词"强命中"所需的最少命中数


@dataclass
class SkillSelection:
    """skill_router 的判定结果。skill_id=None ⇒ 上层回退通用逻辑。"""
    skill_id: Optional[str]
    confidence: float          # 0~1, 强匹配置信度
    reason: str                # 简短结论
    thinking: str              # 给用户展示的判定思路 (思考过程)


_ROUTER_SYSTEM = (
    "你是研究任务类型分类器, 需要谨慎判断, 宁缺毋滥。\n"
    "规则:\n"
    "1) 只有当用户问题与某个候选任务类型【明确、强匹配】(问题的核心诉求正好就是该任务类型在做的事) 时, 才选它;\n"
    "2) 只要稍有牵强、模棱两可、或只是泛泛而问, 一律选 none —— 让系统走通用流程, 不要勉强套用;\n"
    "3) 先用一句话给出判断依据, 再给结论与置信度。\n"
    "严格只输出一个 JSON 对象 (不要 Markdown 代码块、不要多余文字):\n"
    '{"reason": "<一句话中文判断依据>", "skill": "<任务类型id 或 none>", "confidence": <0到1之间的小数>}'
)


def _heuristic_scan(
    query: str, skills: Dict[str, ResearchSkill],
) -> List[Tuple[str, int, List[str]]]:
    """对每个 skill 统计触发词命中, 按 (命中数, priority) 降序返回 [(id, hits, 命中词)]。"""
    q = query or ""
    rows: List[Tuple[str, int, List[str], int]] = []
    for s in skills.values():
        matched: List[str] = []
        for trig in s.triggers:
            t = str(trig)
            try:
                if re.search(t, q, re.I):
                    matched.append(t)
            except re.error:
                if t.lower() in q.lower():
                    matched.append(t)
        if matched:
            rows.append((s.id, len(matched), matched, s.priority))
    rows.sort(key=lambda r: (r[1], r[3]), reverse=True)
    return [(r[0], r[1], r[2]) for r in rows]


def _parse_router_json(
    text: str, known_ids: List[str],
) -> Tuple[Optional[str], float, str]:
    """从模型输出里抽取 {skill, confidence, reason}; 解析失败返回 (None, 0, "")。"""
    candidates = re.findall(r"\{[^{}]*\}", text or "", re.S)
    for c in reversed(candidates):   # 最后一个 JSON 通常是思考后的最终结论
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict) or "skill" not in obj:
            continue
        sid_raw = str(obj.get("skill") or "").strip().lower()
        reason = str(obj.get("reason") or "").strip()
        try:
            conf = float(obj.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        sid: Optional[str] = None
        if sid_raw and sid_raw != "none":
            for kid in known_ids:
                if kid.lower() == sid_raw or kid.lower() in sid_raw:
                    sid = kid
                    break
        return sid, conf, reason
    return None, 0.0, ""


def _llm_select_reasoned(
    llm: Any, query: str, skills: Dict[str, ResearchSkill],
    *, hint: str, prev_skill_id: Optional[str],
    max_tokens: int, disable_thinking: bool, correlation_id: str,
) -> Tuple[Optional[str], float, str]:
    """思考型分类: 让模型推理并输出 JSON {skill, confidence, reason}。"""
    listing = "\n".join(
        f"- {s.id} ({s.name}): {(s.description or '').splitlines()[0] if s.description else ''}"
        for s in skills.values()
    )
    extra = ""
    if hint:
        extra += f"\n关键词初筛倾向 (仅供参考, 不一定准): {hint}"
    if prev_skill_id and prev_skill_id in skills:
        extra += f"\n上一轮采用的任务类型: {prev_skill_id} (若本轮问题确实仍属此类型可延续, 否则照常判 none)"
    user = (
        f"用户研究型问题: {query}\n\n候选任务类型:\n{listing}{extra}\n\n"
        "请判断该问题是否强匹配其中某一个任务类型, 按要求只输出 JSON。"
    )
    try:
        resp = llm.chat(
            system=_ROUTER_SYSTEM, user=user,
            temperature=0.0, max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
    except Exception as e:
        logger.warning(f"[{correlation_id}] [skill_router] LLM 调用失败: {e}")
        return None, 0.0, ""
    return _parse_router_json(str(resp.get("answer") or ""), list(skills.keys()))


def select_skill(
    llm: Any,
    query: str,
    skills: Dict[str, ResearchSkill],
    *,
    mode: str = "hybrid",
    prev_skill_id: Optional[str] = None,
    router_max_tokens: int = 256,
    disable_thinking: bool = False,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    strong_min_hits: int = DEFAULT_STRONG_MIN_HITS,
    correlation_id: str = "-",
) -> SkillSelection:
    """选 skill。仅在【强匹配】时返回具体 skill, 否则返回 skill_id=None (回退通用)。

    mode: off=不选(始终通用) | heuristic=仅触发词强命中 | llm=仅思考模型 | hybrid=思考模型为主、触发词强命中兜底。
    强匹配判定:
      - 思考模型给出某 skill 且 confidence ≥ min_confidence; 或
      - 触发词命中数 ≥ strong_min_hits (关键词强命中)。
    单个触发词命中 / 模型低置信 / 模棱两可 → 一律 None, 走通用逻辑。
    """
    if not skills or str(mode).lower() == "off":
        return SkillSelection(None, 0.0, "skills 关闭", "技能系统未启用，按通用研究方式处理。")
    mode = str(mode).lower()

    scan = _heuristic_scan(query, skills)
    top = scan[0] if scan else None
    top_id = top[0] if top else None
    top_hits = top[1] if top else 0
    top_terms = top[2] if top else []
    hint = (
        f"{top_id}（命中触发词 {('、'.join(top_terms))}）" if top and top_hits else "无关键词命中"
    )
    strong_kw = bool(top and top_hits >= strong_min_hits)

    # 纯触发词模式: 只认强命中
    if mode == "heuristic":
        if strong_kw:
            conf = min(1.0, 0.5 + 0.2 * top_hits)
            return SkillSelection(
                top_id, conf, f"触发词强命中×{top_hits}",
                f"用户问题强命中「{skills[top_id].name}」的触发词（{'、'.join(top_terms)}），判定为该任务类型。",
            )
        return SkillSelection(
            None, 0.0, "触发词未强命中",
            f"仅{('命中 ' + '、'.join(top_terms)) if top_terms else '无关键词命中'}，"
            "不足以强匹配任何专门技能，按通用研究方式处理。",
        )

    # llm / hybrid: 思考模型为决策主体
    if llm is not None:
        sid, conf, reason = _llm_select_reasoned(
            llm, query, skills, hint=hint, prev_skill_id=prev_skill_id,
            max_tokens=router_max_tokens, disable_thinking=disable_thinking,
            correlation_id=correlation_id,
        )
        reason_disp = reason or "（模型未给出依据）"
        if sid and conf >= min_confidence:
            return SkillSelection(
                sid, conf, "思考模型强匹配",
                f"{reason_disp}（判定为「{skills[sid].name}」，置信度 {conf:.0%}）。",
            )
        # 模型不确定, 但触发词强命中 → 用关键词兜底 (仍属强匹配)
        if mode == "hybrid" and strong_kw:
            kw_conf = min(1.0, 0.5 + 0.2 * top_hits)
            return SkillSelection(
                top_id, kw_conf, "触发词强命中兜底",
                f"模型未给出高置信判断，但问题强命中「{skills[top_id].name}」的触发词"
                f"（{'、'.join(top_terms)}），按该任务类型处理。",
            )
        return SkillSelection(
            None, conf, "未强匹配",
            f"{reason_disp} 未达到强匹配标准，按通用研究方式处理。",
        )

    # 无可用 LLM 的 hybrid → 退化为触发词强命中
    if strong_kw:
        conf = min(1.0, 0.5 + 0.2 * top_hits)
        return SkillSelection(
            top_id, conf, f"触发词强命中×{top_hits}",
            f"用户问题强命中「{skills[top_id].name}」的触发词（{'、'.join(top_terms)}），判定为该任务类型。",
        )
    return SkillSelection(
        None, 0.0, "未强匹配",
        "未强匹配到任何专门技能，按通用研究方式处理。",
    )


# ---------------------------------------------------------------------------
# 守卫 (软引导): 把"充分性未满足项"作为额外观测行注入 policy
# ---------------------------------------------------------------------------

_NUM_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|‰|°c|k|μm|um|nm|mm|cm|m|mol|wt%|at%|"
    r"mpa|gpa|kpa|pa|v|mv|μa|ua|ma|a|h|min|s|day|年|天|小时|分钟)",
    re.I,
)


def evaluate_guards(
    skill: ResearchSkill, *, doc_count: int, evidence_texts: List[str],
) -> List[str]:
    """返回当前未满足的守卫项 (人类可读)。空 = 守卫均满足/无可判定守卫。"""
    unmet: List[str] = []
    for g in skill.guards:
        if g == "min_docs":
            need = skill.default_sufficiency.get("min_docs")
            try:
                need = int(need) if need is not None else None
            except (TypeError, ValueError):
                need = None
            if need and doc_count < need:
                unmet.append(f"证据文献数不足 (当前 {doc_count} / 目标 {need})")
        elif g == "need_quantitative":
            if not any(_NUM_UNIT_RE.search(t or "") for t in evidence_texts):
                unmet.append("尚未检索到定量数据 (数值+单位), 需定向补充")
        # per_object_evidence / causal_chain_evidence: 语义守卫, 由 policy.md 提示词引导
    return unmet


# ---------------------------------------------------------------------------
# 目录解析 / 配置 (供 flow 与 API 共用, 避免重复逻辑)
# ---------------------------------------------------------------------------

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pipeline
_REPO_ROOT = os.path.dirname(_PKG_DIR)
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_VALID_PATHS = ("summary", "progressive", "local", "metadata")
_VALID_GUARDS = ("min_docs", "need_quantitative", "per_object_evidence", "causal_chain_evidence")


def resolve_skill_dir(d: str) -> str:
    """把配置里的 skill 目录解析成绝对路径 (不受进程 cwd 影响)。

    - 绝对路径 / ~ 开头: 原样 (展开 ~);
    - 'pipeline/...': 锚定到本 pipeline 包目录 (内置 skill);
    - 其它相对路径: 锚定到仓库根 (如 uploads/skills)。
    """
    d = str(d)
    if os.path.isabs(d) or d.startswith("~"):
        return os.path.expanduser(d)
    if d == "pipeline" or d.startswith("pipeline/"):
        return os.path.join(_PKG_DIR, os.path.relpath(d, "pipeline"))
    return os.path.join(_REPO_ROOT, d)


def resolve_skills_config(skills_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """解析 professional.skills 配置 → enabled / dirs(已解析,含 upload_dir) / upload_dir / router。"""
    cfg = skills_cfg or {}
    raw_dirs = cfg.get("dirs") or ["pipeline/skills"]
    upload_dir = resolve_skill_dir(cfg.get("upload_dir") or "uploads/skills")
    dirs = [resolve_skill_dir(d) for d in raw_dirs]
    if upload_dir not in dirs:
        dirs.append(upload_dir)   # 用户上传目录最后加载 → 同 id 覆盖内置
    router = cfg.get("router", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "dirs": dirs,
        "upload_dir": upload_dir,
        "router_mode": str(router.get("mode", "hybrid")),
        "router_max_tokens": int(router.get("max_tokens", 512)),
        # 选 skill 走"思考模型": 默认开思考 (disable_thinking=False)
        "router_disable_thinking": bool(router.get("disable_thinking", False)),
        "router_min_confidence": float(router.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        "router_strong_min_hits": int(router.get("strong_min_hits", DEFAULT_STRONG_MIN_HITS)),
    }


# ---------------------------------------------------------------------------
# 增删改 (供上传 UI / API 使用)
# ---------------------------------------------------------------------------

def validate_skill_spec(spec: Dict[str, Any]) -> str:
    """校验上传的 skill 规格; 返回错误信息 (空串=通过)。"""
    sid = str(spec.get("id") or "").strip()
    if not SKILL_ID_RE.match(sid):
        return "id 必须是小写字母开头, 仅含小写字母/数字/下划线, 长度 2-41"
    if not str(spec.get("name") or "").strip():
        return "name (技能名) 不能为空"
    if not str(spec.get("plan") or "").strip():
        return "plan (规划提示词) 不能为空"
    if not str(spec.get("policy") or "").strip():
        return "policy (策略提示词) 不能为空"
    for g in spec.get("guards") or []:
        if g not in _VALID_GUARDS:
            return f"未知守卫: {g} (可选: {', '.join(_VALID_GUARDS)})"
    for p in spec.get("prefer_first_paths") or []:
        if p not in _VALID_PATHS:
            return f"未知检索路径: {p} (可选: {', '.join(_VALID_PATHS)})"
    return ""


def _dump_frontmatter(meta: Dict[str, Any]) -> str:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml 未安装, 无法写入 skill")
    body = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).rstrip()
    return f"---\n{body}\n---\n"


def write_skill(upload_dir: str, spec: Dict[str, Any]) -> str:
    """把 skill 规格写成文件夹 (SKILL.md/plan.md/policy.md/synthesis.md)。返回目录路径。"""
    err = validate_skill_spec(spec)
    if err:
        raise ValueError(err)
    sid = str(spec["id"]).strip()
    d = os.path.join(upload_dir, sid)
    os.makedirs(d, exist_ok=True)

    meta: Dict[str, Any] = {"id": sid, "name": str(spec["name"]).strip()}
    if spec.get("priority") is not None:
        meta["priority"] = int(spec["priority"])
    triggers = [str(t).strip() for t in (spec.get("triggers") or []) if str(t).strip()]
    if triggers:
        meta["triggers"] = triggers
    suff = {k: v for k, v in (spec.get("sufficiency") or {}).items() if v not in (None, "", [])}
    if suff:
        meta["sufficiency"] = suff
    paths = [p for p in (spec.get("prefer_first_paths") or []) if p in _VALID_PATHS]
    if paths:
        meta["prefer_first_paths"] = paths
    tuning = {k: v for k, v in (spec.get("tuning") or {}).items() if v not in (None, "")}
    if tuning:
        meta["tuning"] = tuning
    guards = [g for g in (spec.get("guards") or []) if g in _VALID_GUARDS]
    if guards:
        meta["guards"] = guards

    description = str(spec.get("description") or "").strip()
    skill_md = _dump_frontmatter(meta) + "\n## Description\n\n" + (description or sid) + "\n"
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(skill_md)
    with open(os.path.join(d, "plan.md"), "w", encoding="utf-8") as f:
        f.write(str(spec["plan"]).strip() + "\n")
    with open(os.path.join(d, "policy.md"), "w", encoding="utf-8") as f:
        f.write(str(spec["policy"]).strip() + "\n")

    syn_sys = str(spec.get("synthesis_system") or "").strip()
    syn_think = str(spec.get("synthesis_thinking") or "").strip()
    syn_user = str(spec.get("synthesis_user") or "").strip()
    synth_path = os.path.join(d, "synthesis.md")
    if syn_sys or syn_think or syn_user:
        parts: List[str] = []
        if syn_sys:
            parts.append("## System\n\n" + syn_sys)
        if syn_think:
            parts.append("## Thinking\n\n" + syn_think)
        if syn_user:
            parts.append("## User\n\n" + syn_user)
        with open(synth_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts) + "\n")
    elif os.path.isfile(synth_path):
        os.remove(synth_path)   # 编辑时清空了综述段 → 删除旧文件, 回退通用模板

    logger.info(f"[skills] 写入 skill: {sid} → {d}")
    return d


def delete_skill(upload_dir: str, skill_id: str) -> bool:
    """删除上传目录下的某个 skill 文件夹 (内置 skill 不在此目录, 不受影响)。"""
    sid = str(skill_id).strip()
    if not SKILL_ID_RE.match(sid):
        raise ValueError("非法 skill id")
    d = os.path.join(upload_dir, sid)
    # 防目录穿越: 解析后必须仍在 upload_dir 内
    if os.path.realpath(d).startswith(os.path.realpath(upload_dir) + os.sep) is False:
        raise ValueError("非法路径")
    if os.path.isdir(d):
        shutil.rmtree(d)
        logger.info(f"[skills] 删除 skill: {sid}")
        return True
    return False


def skill_to_summary(skill: ResearchSkill, *, upload_dir: str) -> Dict[str, Any]:
    """把 ResearchSkill 转成给前端列表/编辑用的 dict (含可编辑的提示词正文)。"""
    editable = bool(
        skill.source_dir
        and os.path.realpath(skill.source_dir).startswith(os.path.realpath(upload_dir))
    )
    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "priority": skill.priority,
        "triggers": skill.triggers,
        "sufficiency": skill.default_sufficiency,
        "prefer_first_paths": skill.prefer_first_paths,
        "tuning": {
            "max_rounds": skill.max_rounds,
            "max_batches": skill.max_batches,
            "gap_stall_limit": skill.gap_stall_limit,
            "stall_quality_floor": skill.stall_quality_floor,
        },
        "guards": skill.guards,
        "plan": skill.plan_system,
        "policy": skill.policy_system,
        "synthesis_system": skill.synthesis_system,
        "synthesis_thinking": skill.synthesis_thinking_system,
        "synthesis_user": skill.synthesis_user_template,
        "editable": editable,
    }


def skill_template() -> Dict[str, Any]:
    """返回新建 skill 的填写模版 (字段说明 + 示例), 供前端表单渲染。"""
    return {
        "fields": [
            {"key": "id", "label": "技能 ID", "type": "text", "required": True,
             "help": "小写字母开头, 仅含小写字母/数字/下划线 (如 quantitative_extraction)"},
            {"key": "name", "label": "技能名称", "type": "text", "required": True,
             "help": "中文展示名 (如 定量数据抽取)"},
            {"key": "description", "label": "适用场景说明", "type": "textarea", "required": False,
             "help": "什么样的用户发话该用这个技能 — 给分类器判断用"},
            {"key": "priority", "label": "优先级", "type": "number", "required": False,
             "help": "多个技能触发词同时命中时, 数值大者优先 (默认 50)"},
            {"key": "triggers", "label": "触发词", "type": "list", "required": False,
             "help": "命中用户发话即倾向选该技能; 支持正则。每行一个"},
            {"key": "sufficiency", "label": "收口标准", "type": "sufficiency", "required": False,
             "help": "min_docs(至少文献数) / need_conflict_check / need_quantitative_data"},
            {"key": "tuning", "label": "调参", "type": "tuning", "required": False,
             "help": "max_rounds / max_batches / gap_stall_limit (留空用全局默认)"},
            {"key": "guards", "label": "守卫", "type": "guards", "required": False,
             "help": "min_docs / need_quantitative / per_object_evidence / causal_chain_evidence"},
            {"key": "plan", "label": "规划提示词 (plan)", "type": "textarea", "required": True,
             "help": "选中 research_plan 后如何拆解 facets 与首轮批次 (替换通用拆解段)"},
            {"key": "policy", "label": "策略提示词 (policy)", "type": "textarea", "required": True,
             "help": "每轮检索后如何判断 继续/收口/反问 并引导下一步 (替换通用判断段)"},
            {"key": "synthesis_system", "label": "综述结构提示词", "type": "textarea", "required": False,
             "help": "可选: 最终综述的输出结构; 留空回退通用综述模板"},
            {"key": "synthesis_thinking", "label": "综述分析思路提示词", "type": "textarea", "required": False,
             "help": "可选: 综述前的中文分析思路 (思考过程)"},
            {"key": "synthesis_user", "label": "综述 User 模版", "type": "textarea", "required": False,
             "help": "可选: 须包含 {context} 占位符"},
        ],
        "valid_paths": list(_VALID_PATHS),
        "valid_guards": list(_VALID_GUARDS),
        "example": {
            "id": "quantitative_extraction",
            "name": "定量数据抽取",
            "description": "当用户需要具体数值/速率/含量/电化学参数等定量证据时使用。",
            "priority": 58,
            "triggers": ["数值", "多少", "速率", "含量", "参数", "定量"],
            "sufficiency": {"min_docs": 2, "need_quantitative_data": True},
            "tuning": {"max_rounds": 5},
            "guards": ["need_quantitative"],
            "plan": (
                "选择 research_plan 时（定量抽取任务）：把问题拆成需要定量证据的若干指标维度，"
                "每个 facet 聚焦一个可量化指标。首轮可用 summary 定位文献，随后用 progressive/local "
                "深入正文与表格抽取数值。关键词带上指标名与单位线索（含英文同义词）。"
            ),
            "policy": (
                "判断依据（定量抽取任务）：核心是拿到带数值+单位的硬证据。若本轮仍只有定性描述、"
                "缺关键指标数值，则 continue 并定向补该指标；各关键指标都已拿到可引用的定量数据后 finish；"
                "若语料确实无定量数据则 clarify 说明。"
            ),
            "synthesis_system": "",
            "synthesis_thinking": "",
            "synthesis_user": "",
        },
    }
