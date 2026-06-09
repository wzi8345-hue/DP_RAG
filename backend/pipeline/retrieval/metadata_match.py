"""Metadata 检索文本匹配 (#11 图表标签 + #12 实体大小写)。

统一生成 Milvus LIKE 子句与 Python 侧评分规则, 避免 filter 阶段漏候选。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from .retrievers import _escape_like

_DEFAULT_MAX_LIKE_PER_LABEL = 12
_DEFAULT_MAX_ENTITY_VARIANTS = 3


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _like_clause(fragment: str) -> str:
    return f'content like "%{fragment}%"'


def _parse_fig_numeric_suffix(label: str) -> Tuple[str, str]:
    """解析数字型 fig 编号: "3a" → ("3", "a"); 纯数字 → suffix 为空。"""
    raw = (label or "").strip()
    if re.fullmatch(r"[Ss]\d+", raw):
        return raw, ""
    m = re.fullmatch(r"(\d+)([a-zA-Z])?", raw)
    if m:
        return m.group(1), (m.group(2) or "").lower()
    return raw, ""


def _fig_paren_like_fragments(base: str, suffix: str) -> List[str]:
    """论文常见 Fig. 3(a) 括号小写变体 (精简 prefix, 控制 clause 数量)。"""
    if not suffix or not base.isdigit():
        return []
    su, sU = suffix.lower(), suffix.upper()
    frags: List[str] = []
    for prefix in ("Fig. ", "Fig ", "Figure "):
        frags.append(f"{prefix}{base}({su})")
        if su != sU:
            frags.append(f"{prefix}{base}({sU})")
    frags.append(f"图{base}({su})")
    frags.append(f"图 {base}({su})")
    return frags


def fig_like_clauses(label: str, *, max_clauses: int = _DEFAULT_MAX_LIKE_PER_LABEL) -> List[str]:
    """为单个 fig 编号生成 Milvus LIKE 子句 (上限 max_clauses)。"""
    raw = (label or "").strip()
    if not raw:
        return []

    esc = _escape_like(raw)
    base, suffix = _parse_fig_numeric_suffix(raw)
    paren_clauses: List[str] = []
    if suffix and base.isdigit():
        for frag in _fig_paren_like_fragments(base, suffix):
            paren_clauses.append(_like_clause(_escape_like(frag)))

    std_clauses: List[str] = []
    for prefix in ("Fig.", "Fig", "Figure", "figure"):
        std_clauses.append(_like_clause(f"{prefix} {esc}"))
        std_clauses.append(_like_clause(f"{prefix}{esc}"))

    std_clauses.append(_like_clause(f"图 {esc}"))
    std_clauses.append(_like_clause(f"图{esc}"))

    if raw.isdigit():
        for prefix in ("Fig.S", "Fig. S", "Fig S"):
            std_clauses.append(_like_clause(f"{prefix}{esc}"))
            std_clauses.append(_like_clause(f"{prefix} {esc}"))

    if re.fullmatch(r"[Ss]\d+", raw):
        num = raw[1:]
        esc_num = _escape_like(num)
        for prefix in ("Fig.S", "Fig. S", "Fig S"):
            std_clauses.append(_like_clause(f"{prefix}{esc_num}"))
            std_clauses.append(_like_clause(f"{prefix} {esc_num}"))

    if suffix and base.isdigit():
        # 括号变体优先, 再保留直接写法 (Fig. 3a / Fig.3a)
        essential = [
            _like_clause(f"Fig. {esc}"),
            _like_clause(f"Fig.{esc}"),
            _like_clause(f"Figure {esc}"),
            _like_clause(f"图 {esc}"),
        ]
        clauses = _dedupe_preserve_order(paren_clauses + essential + std_clauses)
    else:
        clauses = _dedupe_preserve_order(paren_clauses + std_clauses)

    return clauses[:max_clauses]


def table_like_clauses(label: str, *, max_clauses: int = _DEFAULT_MAX_LIKE_PER_LABEL) -> List[str]:
    """为单个 table 编号生成 Milvus LIKE 子句。"""
    raw = (label or "").strip()
    if not raw:
        return []

    esc = _escape_like(raw)
    clauses: List[str] = []

    for prefix in ("Table", "table", "TABLE"):
        clauses.append(_like_clause(f"{prefix} {esc}"))
        clauses.append(_like_clause(f"{prefix}{esc}"))

    clauses.append(_like_clause(f"表 {esc}"))
    clauses.append(_like_clause(f"表{esc}"))

    return _dedupe_preserve_order(clauses)[:max_clauses]


def collect_ref_like_clauses(
    fig_refs: Sequence[str],
    table_refs: Sequence[str],
) -> List[str]:
    """合并 fig/table 全部 LIKE 子句。"""
    clauses: List[str] = []
    for label in fig_refs:
        clauses.extend(fig_like_clauses(label))
    for label in table_refs:
        clauses.extend(table_like_clauses(label))
    return _dedupe_preserve_order(clauses)


def _fig_paren_regexes(base: str, suffix: str) -> List[str]:
    """Fig. 3(a) / Fig. 3(A) 括号变体 regex。"""
    if not suffix or not base.isdigit():
        return []
    b_esc = re.escape(base)
    su, sU = re.escape(suffix.lower()), re.escape(suffix.upper())
    char_cls = f"[{su}{sU}]" if su != sU else su
    return [
        rf"\bFig(?:ure|\.)?\s*{b_esc}\s*\({char_cls}\)(?![0-9A-Za-z])",
        rf"Fig\.{b_esc}\({char_cls}\)(?![0-9A-Za-z])",
        rf"图\s*{b_esc}\s*\({char_cls}\)(?![0-9A-Za-z])",
    ]


def _fig_label_regexes(label: str) -> List[re.Pattern]:
    raw = (label or "").strip()
    if not raw:
        return []

    esc = re.escape(raw)
    base, suffix = _parse_fig_numeric_suffix(raw)
    patterns = [
        rf"\bFig(?:ure|\.)?\s*{esc}(?![0-9A-Za-z])",
        rf"Fig\.{esc}(?![0-9A-Za-z])",
        rf"图\s*{esc}(?![0-9A-Za-z])",
    ]
    patterns.extend(_fig_paren_regexes(base, suffix))
    if raw.isdigit():
        num_esc = re.escape(raw)
        patterns.extend([
            rf"\bFig\.?\s*S{num_esc}(?![0-9A-Za-z])",
            rf"Fig\.S{num_esc}(?![0-9A-Za-z])",
        ])
    if re.fullmatch(r"[Ss]\d+", raw):
        num_esc = re.escape(raw[1:])
        patterns.extend([
            rf"\bFig\.?\s*S{num_esc}(?![0-9A-Za-z])",
            rf"Fig\.S{num_esc}(?![0-9A-Za-z])",
        ])
    return [re.compile(p, re.IGNORECASE) for p in _dedupe_preserve_order(patterns)]


def _table_label_regexes(label: str) -> List[re.Pattern]:
    raw = (label or "").strip()
    if not raw:
        return []

    esc = re.escape(raw)
    patterns = [
        rf"\bTable\s*{esc}(?![0-9A-Za-z])",
        rf"Table{esc}(?![0-9A-Za-z])",
        rf"表\s*{esc}(?![0-9A-Za-z])",
    ]
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def score_fig_table_refs(
    blob: str,
    fig_refs: Sequence[str],
    table_refs: Sequence[str],
    hit_type: str,
) -> Tuple[float, List[str]]:
    """对 content blob 评分; 返回 (score, matched_labels)。"""
    score = 0.0
    matched: List[str] = []

    for label in fig_refs:
        if any(p.search(blob) for p in _fig_label_regexes(label)):
            matched.append(f"Fig.{label}")
            score += 5.0 if hit_type == "image" else 2.0

    for label in table_refs:
        if any(p.search(blob) for p in _table_label_regexes(label)):
            matched.append(f"Table {label}")
            score += 5.0 if hit_type == "table" else 2.0

    return score, matched


def entity_case_variants(entity: str, *, max_variants: int = _DEFAULT_MAX_ENTITY_VARIANTS) -> List[str]:
    """实体 LIKE 变体: original + lower + upper (去重, 上限 max_variants)。"""
    e = (entity or "").strip()
    if not e:
        return []
    return _dedupe_preserve_order([e, e.lower(), e.upper()])[:max_variants]


def entity_like_clauses(entity: str) -> List[str]:
    """单个 entity 的大小写不敏感 Milvus LIKE 子句。"""
    return [
        f'content like "%{_escape_like(v)}%"'
        for v in entity_case_variants(entity)
    ]


def collect_entity_like_clauses(entities: Sequence[str]) -> List[str]:
    """多个 entity 的全部 LIKE 子句 (entity 间 OR, 变体间 OR)。"""
    clauses: List[str] = []
    for entity in entities:
        e = (entity or "").strip()
        if e:
            clauses.extend(entity_like_clauses(e))
    return _dedupe_preserve_order(clauses)
