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
    # 自描述段 (供 LLM 路由判断, 取代触发词关键字匹配)
    when_to_use: str = ""
    when_not_to_use: str = ""
    examples: List[str] = field(default_factory=list)       # 正例: 应命中本技能的问题
    anti_examples: List[str] = field(default_factory=list)  # 反例: 不应命中 (易混)
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


def _parse_bullets(text: str) -> List[str]:
    """把 markdown 无序/有序列表正文解析成条目列表 (剥掉 -/*/数字. 前缀)。"""
    items: List[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", s)
        items.append(m.group(1).strip() if m else s)
    return [i for i in items if i]


# SKILL.md 自描述段的标题别名 (英文/中文都认), 统一映射到字段
_SECTION_ALIASES = {
    "when_to_use": ("when to use", "适用场景", "何时使用", "适用"),
    "when_not_to_use": ("when not to use", "不适用场景", "何时不用", "不适用"),
    "examples": ("examples", "正例", "命中示例", "示例"),
    "anti_examples": ("anti-examples", "anti examples", "反例", "不命中示例"),
}


def _section(sections: Dict[str, str], field_key: str) -> str:
    for alias in _SECTION_ALIASES.get(field_key, ()):  # noqa: B007
        if alias in sections:
            return sections[alias]
    return ""


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
    when_to_use = _section(desc_sections, "when_to_use")
    when_not_to_use = _section(desc_sections, "when_not_to_use")
    examples = _parse_bullets(_section(desc_sections, "examples"))
    anti_examples = _parse_bullets(_section(desc_sections, "anti_examples"))

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
        when_to_use=str(when_to_use or "").strip(),
        when_not_to_use=str(when_not_to_use or "").strip(),
        examples=examples,
        anti_examples=anti_examples,
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
    "1) 只有当用户问题与某个候选任务类型【明确、强匹配】(问题的核心诉求正好就是该任务类型在做的事, "
    "且符合其「适用」、贴近其「正例」) 时, 才选它;\n"
    "2) 若问题落入某类型的「不适用」或更像其「反例」, 不要选它;\n"
    "3) 只要稍有牵强、模棱两可、或只是泛泛而问, 一律选 none —— 让系统走通用流程, 不要勉强套用;\n"
    "4) 先用一句话给出判断依据, 再给结论与置信度。\n"
    "严格只输出一个 JSON 对象 (不要 Markdown 代码块、不要多余文字):\n"
    '{"reason": "<一句话中文判断依据>", "skill": "<任务类型id 或 none>", "confidence": <0到1之间的小数>}'
)


def _format_candidate(s: ResearchSkill) -> str:
    """把单个 skill 渲染成候选说明块 (适用/不适用/正例/反例), 供分类模型判断。"""
    lines = [f"### {s.id}（{s.name}）"]
    head = s.when_to_use or ((s.description or "").splitlines()[0] if s.description else "")
    if head:
        lines.append(f"适用: {head}")
    if s.when_not_to_use:
        lines.append(f"不适用: {s.when_not_to_use}")
    if s.examples:
        lines.append("正例:")
        lines += [f"  - {e}" for e in s.examples[:5]]
    if s.anti_examples:
        lines.append("反例:")
        lines += [f"  - {e}" for e in s.anti_examples[:5]]
    return "\n".join(lines)


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
    *, prev_skill_id: Optional[str],
    max_tokens: int, disable_thinking: bool, correlation_id: str,
) -> Tuple[Optional[str], float, str]:
    """思考型分类: 让模型推理并输出 JSON {skill, confidence, reason}。

    候选清单含每个技能的「适用/不适用/正例/反例」自描述, 不再依赖关键词触发。
    """
    listing = "\n\n".join(_format_candidate(s) for s in skills.values())
    extra = ""
    if prev_skill_id and prev_skill_id in skills:
        extra += f"\n\n上一轮采用的任务类型: {prev_skill_id} (若本轮问题确实仍属此类型可延续, 否则照常判 none)"
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
    mode: str = "llm",
    prev_skill_id: Optional[str] = None,
    router_max_tokens: int = 256,
    disable_thinking: bool = False,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    correlation_id: str = "-",
) -> SkillSelection:
    """选 skill。仅在【强匹配】时返回具体 skill, 否则返回 skill_id=None (回退通用)。

    判定完全交给思考模型: 依据每个技能的「适用/不适用/正例/反例」自描述判断,
    模型给出某 skill 且 confidence ≥ min_confidence 才算强匹配; 模棱两可一律 None。

    mode: off=不选(始终通用) | 其它(llm)=思考模型判断。
    无可用 LLM 时无法判断 → 一律 None (回退通用)。
    """
    if not skills or str(mode).lower() == "off":
        return SkillSelection(None, 0.0, "skills 关闭", "技能系统未启用，按通用研究方式处理。")

    if llm is None:
        return SkillSelection(
            None, 0.0, "无可用分类模型",
            "未配置分类模型，无法判断任务类型，按通用研究方式处理。",
        )

    sid, conf, reason = _llm_select_reasoned(
        llm, query, skills, prev_skill_id=prev_skill_id,
        max_tokens=router_max_tokens, disable_thinking=disable_thinking,
        correlation_id=correlation_id,
    )
    reason_disp = reason or "（模型未给出依据）"
    if sid and conf >= min_confidence:
        return SkillSelection(
            sid, conf, "思考模型强匹配",
            f"{reason_disp}（判定为「{skills[sid].name}」，置信度 {conf:.0%}）。",
        )
    return SkillSelection(
        None, conf, "未强匹配",
        f"{reason_disp} 未达到强匹配标准，按通用研究方式处理。",
    )


# ---------------------------------------------------------------------------
# 守卫 (软引导): 把"充分性未满足项"作为额外观测行注入 policy
# ---------------------------------------------------------------------------

_NUM_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|‰|°c|k|μm|um|nm|mm|cm|m|mol|wt%|at%|"
    r"mpa|gpa|kpa|pa|v|mv|μa|ua|ma|a|h|min|s|day|年|天|小时|分钟)",
    re.I,
)

# 因果/机理连接词: 证据里出现才算讲清了"原因→过程→结果", 否则多为现象描述。
_CAUSAL_RE = re.compile(
    r"(因为|由于|导致|致使|使得|从而|因而|进而|引起|造成|促使|归因于|"
    r"机理|机制|反应|作用机制|之所以|原因在于|"
    r"because|due to|leads? to|result(?:s|ed)? in|cause[sd]?|owing to|mechanism)",
    re.I,
)


def _covered_facet(facet_id: str, covered: List[str]) -> bool:
    """facet 是否已被判定覆盖 (covered 来自 policy 的 covered 维度, 做双向包含匹配)。"""
    fid = facet_id.lower()
    return any(fid in c.lower() or c.lower() in fid for c in covered if c)


def evaluate_guards(
    skill: ResearchSkill,
    *,
    doc_count: int,
    evidence_texts: List[str],
    facet_ids: Optional[List[str]] = None,
    covered: Optional[List[str]] = None,
) -> List[str]:
    """返回当前未满足的守卫项 (人类可读)。空 = 守卫均满足/无可判定守卫。

    facet_ids / covered 供 per_object_evidence 判定"每个规划维度都已获证据覆盖"。
    """
    facet_ids = facet_ids or []
    covered = covered or []
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
        elif g == "per_object_evidence":
            # 对称覆盖: 规划出的每个维度 (对比任务里 = 每个被比较对象) 都应有证据
            missing = [f for f in facet_ids if not _covered_facet(f, covered)]
            if missing:
                unmet.append(
                    "以下规划维度尚未获得证据覆盖 (对比需对称覆盖各对象再收口): "
                    + ", ".join(missing[:6])
                )
        elif g == "causal_chain_evidence":
            # 因果链: 证据需含原因/过程/机理语句, 而非纯现象描述
            if evidence_texts and not any(_CAUSAL_RE.search(t or "") for t in evidence_texts):
                unmet.append("证据偏现象描述, 尚缺因果/机理链条 (原因→过程→结果) 证据, 需定向补充")
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
        # off=始终通用 | llm(默认)=思考模型按自描述判断 (已无触发词关键字匹配)
        "router_mode": str(router.get("mode", "llm")),
        "router_max_tokens": int(router.get("max_tokens", 512)),
        # 默认关思考: 开思考时 vLLM 把内容放 reasoning 通道, answer 常为空导致路由恒 none。
        # 关思考后模型直接输出 JSON(含 reason 作为判定思路)。
        "router_disable_thinking": bool(router.get("disable_thinking", True)),
        "router_min_confidence": float(router.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
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
    when_to_use = str(spec.get("when_to_use") or "").strip()
    when_not_to_use = str(spec.get("when_not_to_use") or "").strip()
    examples = [str(e).strip() for e in (spec.get("examples") or []) if str(e).strip()]
    anti_examples = [str(e).strip() for e in (spec.get("anti_examples") or []) if str(e).strip()]

    parts: List[str] = [_dump_frontmatter(meta).rstrip()]
    parts.append("## Description\n\n" + (description or sid))
    if when_to_use:
        parts.append("## When to use\n\n" + when_to_use)
    if when_not_to_use:
        parts.append("## When not to use\n\n" + when_not_to_use)
    if examples:
        parts.append("## Examples\n\n" + "\n".join(f"- {e}" for e in examples))
    if anti_examples:
        parts.append("## Anti-examples\n\n" + "\n".join(f"- {e}" for e in anti_examples))
    skill_md = "\n\n".join(parts) + "\n"
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
        "when_to_use": skill.when_to_use,
        "when_not_to_use": skill.when_not_to_use,
        "examples": skill.examples,
        "anti_examples": skill.anti_examples,
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
            {"key": "description", "label": "一句话说明", "type": "textarea", "required": False,
             "help": "这个技能在做什么 — 一句话概述"},
            {"key": "priority", "label": "优先级", "type": "number", "required": False,
             "help": "极少数模型判定平手时数值大者优先 (默认 50)"},
            {"key": "when_to_use", "label": "适用场景", "type": "textarea", "required": False,
             "help": "什么样的提问该用这个技能 — 分类模型据此判断 (越具体越准)"},
            {"key": "when_not_to_use", "label": "不适用场景", "type": "textarea", "required": False,
             "help": "哪些易混的提问不该用这个技能 (与相邻技能划清边界)"},
            {"key": "examples", "label": "正例", "type": "list", "required": False,
             "help": "应命中本技能的代表性问题, 每行一个"},
            {"key": "anti_examples", "label": "反例", "type": "list", "required": False,
             "help": "看似相关但不该命中的问题 (说明该归哪类), 每行一个"},
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
            "description": "从文献中抽取具体数值/速率/含量/电化学参数等定量证据。",
            "priority": 58,
            "when_to_use": "用户要的是带数值与单位的硬证据 (具体多少、速率、含量、电化学参数), 而非定性结论。",
            "when_not_to_use": "用户只想要机理解释、定性对比或研究综述, 不强调拿到具体数值。",
            "examples": [
                "锌铝镁镀层在中性盐雾下的腐蚀速率是多少",
                "这几篇文献里 Mg 含量的范围是多少",
            ],
            "anti_examples": [
                "锌铝镁的耐蚀机理是什么 (→ 机理分析)",
                "综述一下锌铝镁的研究现状 (→ 文献综述)",
            ],
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
