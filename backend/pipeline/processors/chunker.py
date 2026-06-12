"""知识块构建器: 从 MinerU v2 的 content_list_v2.json 构建 knowledge blocks。

v3 chunking (本次修改) 与 v2 的关键差异:
- **段落级切分**: 不再把一个 title 下的所有段落合并为一个 chunk;
  每个 "逻辑段落" 一个 chunk, 并带 `paragraph_index` (文档内 1-based 序号)。
- **误切合并**: 用句尾标点 (没有 . 。 ! ? ; ;) + 下一段首字符特征 (小写 / 连接词)
  判断 "MinerU 把同一段拆成两段" 的情况, 自动合并。
- **长段落继续走 semantic_split**: 单段超过 max_chars 时仍调用 semantic_splitter
  做语义二次切分, 子 chunk 共享同一 paragraph_index 但 chunk_index 区分。
- **摘要识别只保留强信号**: 仅当 (a) section title 命中 abstract/摘要/summary
  关键字, 或前 2 页出现 "Abstract:"/"摘要：" 显式信号词 + section 段首核验; 或
  (b) section title 与摘要查询词的向量相似度 ≥ 阈值 时, 才判为 summary chunk.
  彻底去掉 "前 N 个 title 当摘要"、"长度 + 句数像摘要" 等弱启发式, 防止把
  introduction / related work 误判为摘要.
- **LLM 兜底摘要**: 上面强信号都没命中 + 传入了 LLMClient 时, 把第一页正文喂给
  LLM 让它合成一段准确摘要 (type=summary, section="[LLM-synthesized abstract]").
- 输出 chunk schema 增加: paragraph_index (文档内 1-based; 非正文型为 -1),
  parent_chunk_id / chunk_index / chunk_total (仅长段切分子块)。
"""

from __future__ import annotations

import glob
import logging
import math
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..clients.embedding import EmbeddingClient
from .semantic_splitter import (
    semantic_split,
    DEFAULT_TARGET_CHARS,
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_BREAKPOINT_PCT,
)

if TYPE_CHECKING:
    from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

NOISE_TYPES = {"page_header", "page_footer", "page_footnote", "page_number"}
SUMMARY_TITLE_COUNT = 1
SUMMARY_SIM_THRESHOLD = 1.4

# ── 期刊元数据行 (CNKI 排版常见): 整段都是这种 label 时直接丢弃, 不进 chunk ──
# 例: "中图分类号：TG172.3", "文献标识码：A", "文章编号：1672-9242(2023)08-0114-08"
#     "DOI：10.7643/issn.1672-9242.2023.08.015", "收稿日期：2023-05-12"
# 这些行做 BM25 时 token 都很 rare (DOI 数字、文章编号、出版年月), 会被 IDF 打高分
# 误命中; 语义上也是无信号. 在 _group_logical_paragraphs 入口直接 drop, 不计入 chunk.
METADATA_LINE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"中\s*图\s*分\s*类\s*号|"
    r"文\s*献\s*标\s*[识志]\s*码|"
    r"文\s*章\s*编\s*号|"
    r"DOI(?:\s*号)?|"
    r"doi|"
    r"收\s*稿\s*日\s*期|"
    r"修\s*回\s*日\s*期|"
    r"基\s*金\s*项\s*目|"
    r"作\s*者\s*简\s*介|"
    r"通\s*[讯信]\s*作\s*者|"
    r"作\s*者\s*单\s*位|"
    r"CLC\s*number|"
    r"document\s*code|"
    r"article\s*id"
    r")\s*[:：]",
    re.IGNORECASE,
)
# 单段最长不超过 120 字符时才视为 metadata 行 (避免误伤含"DOI" 字样的正文段);
# 真实 metadata 行都很短 (典型 10-50 字符), 这个上限留足 margin.
METADATA_LINE_MAX_CHARS = 120


def _is_metadata_line(text: str) -> bool:
    """整段都是期刊元数据行 (中图分类号 / 文献标识码 / 文章编号 / DOI / 收稿日期 等)?

    Examples:
        >>> _is_metadata_line('中图分类号：TG172.3')
        True
        >>> _is_metadata_line('DOI：10.7643/issn.1672-9242.2023.08.015')
        True
        >>> _is_metadata_line('文章编号：1672-9242(2023)08-0114-08')
        True
        >>> _is_metadata_line('本文 DOI 申请由出版社统一处理。')
        False
        >>> _is_metadata_line('耐候钢的腐蚀机理研究')
        False
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > METADATA_LINE_MAX_CHARS:
        return False
    return bool(METADATA_LINE_PREFIX_RE.match(stripped))

# 长 chunk 二次切分参数 (从 default_config.yaml 注入, 这里是兜底默认)
SPLIT_TARGET_CHARS = DEFAULT_TARGET_CHARS
SPLIT_MAX_CHARS = DEFAULT_MAX_CHARS
SPLIT_MIN_CHARS = DEFAULT_MIN_CHARS
SPLIT_BREAKPOINT_PCT = DEFAULT_BREAKPOINT_PCT
# 下列 SUMMARY_* 常量是兜底默认值; 实际运行时优先用 yaml 的 chunking.summary 子表.
# 见 default_config.yaml 中的 chunking.summary 注释了解每级匹配语义.
SUMMARY_QUERY_TEXTS = [
    "abstract 摘要",
    "summary conclusion 总结 结论",
    "overview introduction 综述 引言 概述",
]
SUMMARY_BM25_QUERY_TEXTS = [
    "abstract",
    "summary",
    "overview",
    "executive summary",
    "摘要",
    "概要",
]
SUMMARY_BM25_THRESHOLD = 0.5
SUMMARY_TITLE_PATTERNS = [
    re.compile(r"\babstract\b", re.IGNORECASE),
    re.compile(r"\bexecutive\s+summary\b", re.IGNORECASE),
    re.compile(r"\bsynopsis\b", re.IGNORECASE),
    re.compile(r"\bsummary\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\btl;?\s*dr\b", re.IGNORECASE),
    re.compile(r"摘\s*要"),
    re.compile(r"概\s*要"),
    re.compile(r"提\s*要"),
    re.compile(r"梗\s*概"),
]
SUMMARY_TEXT_PATTERNS = [
    re.compile(r"\babstract\b\s*[:：\-]?", re.IGNORECASE),
    re.compile(r"\bsummary\b\s*[:：\-]?", re.IGNORECASE),
    re.compile(r"\bexecutive\s+summary\b\s*[:：\-]?", re.IGNORECASE),
    re.compile(r"摘\s*要\s*[:：\-]?"),
    re.compile(r"概\s*要\s*[:：\-]?"),
]
SUMMARY_STOP_PATTERNS = [
    re.compile(r"\bkeywords?\b\s*[:：]", re.IGNORECASE),
    re.compile(r"关\s*键\s*词\s*[:：]"),
    re.compile(r"\b\d+(?:\.\d+)*\s+(?:introduction|background)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:引言|前言)\s*$"),
]

# ── spaced-letters normalizer (修 PDF 字间距 artifact) ─────────────────
# 很多 PDF (UniParser 后端尤其常见) 会把 section title 的字间距解析成空格,
# 例如 "A B S T R A C T" / "I N T R O D U C T I O N" / "K E Y W O R D S".
# 这会让 \babstract\b / BM25 token "abstract" 都无法命中, 必须先折叠回 "ABSTRACT"
# 再喂给字面匹配 / BM25 / 向量编码.
#
# 规则: 连续 ≥3 个 "单字母 + 空白" 且后面跟一个单字母, 视为被字间距拆开的词,
# 把它们的空白全部抹掉. 边界用 \b 防止把句子里的 "I a m" 这种正常缩写折叠.
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")
# CJK 字符间距 artifact: '耐 候 钢 和 碳' / '可 以 看 到 ， 耐 候 钢'.
# 规则: ≥3 个 "(CJK 字 / 常见 CJK 标点) + 单空白" + 1 个 CJK 字 / 标点.
# 字符类包含 CJK 表意文字 + 常用全角/中文标点 (逗号 句号 问号 顿号 括号等),
# 避免被中文逗号/句号"断开"成多段 ('可 以 看 到 ，' 后接 '耐 候 钢' 仍应折叠);
# 不包含全角数字 (\uff10-\uff19), 不包含全角字母 (\uff21-\uff3a / \uff41-\uff5a),
# 避免"图 1 表 2"这种被误折叠. \s 用 [ \t] 而非 + 是为了不跨换行合并.
_CJK_RUN_CHAR_CLASS = (
    r"[\u4e00-\u9fff\u3400-\u4dbf"
    r"\u3001\u3002\u300a\u300b\u300c\u300d\u300e\u300f"
    r"\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f]"
)
_SPACED_CJK_RE = re.compile(
    rf"(?:{_CJK_RUN_CHAR_CLASS}[ \t]){{2,}}{_CJK_RUN_CHAR_CLASS}"
)


def _normalize_spaced_letters(s: str) -> str:
    """把 'A B S T R A C T' / 'A B S- T R A C T' 等 PDF 字间距 artifact 折叠成单词.

    Examples:
        >>> _normalize_spaced_letters("A B S T R A C T")
        'ABSTRACT'
        >>> _normalize_spaced_letters("I N T R O D U C T I O N")
        'INTRODUCTION'
        >>> _normalize_spaced_letters("Section 1. Introduction")
        'Section 1. Introduction'
        >>> _normalize_spaced_letters("3.2 Phase transformation")
        '3.2 Phase transformation'
    """
    if not s or len(s) < 5:
        return s
    return _SPACED_LETTERS_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), s)


def _normalize_spaced_cjk(s: str) -> str:
    """折叠 PDF OCR 字间距 artifact: '耐 候 钢 和 碳' -> '耐候钢和碳'.

    只对 ≥3 个 CJK 字符之间被单空白分隔的序列起作用; 不会破坏正常的中文段落
    (正常中文段落字间无空格), 也不会跨换行合并.

    Examples:
        >>> _normalize_spaced_cjk('可 以 看 到 ， 耐 候 钢 和 碳 钢')
        '可以看到，耐候钢和碳钢'
        >>> _normalize_spaced_cjk('耐候钢和碳钢的主要成分')
        '耐候钢和碳钢的主要成分'
        >>> _normalize_spaced_cjk('图 1 所示')
        '图 1 所示'
    """
    if not s or len(s) < 5:
        return s
    return _SPACED_CJK_RE.sub(lambda m: re.sub(r"[ \t]+", "", m.group(0)), s)


# PDF/OCR 常在两字 CJK 标签各字间插入空格 / 零宽字符 / 不换行空格等,
# 例如 "摘 要" / "摘\u200b要". 仅 \s* 匹配不了零宽字符, 需先折叠再喂给摘要检测.
_CJK_LABEL_WS = r"[\s\u200b\u200c\u200d\ufeff\u00a0\u3000\u2060]*"
_CJK_SUMMARY_LABEL_COLLAPSE_RES: List[tuple] = [
    (re.compile("摘" + _CJK_LABEL_WS + "要"), "摘要"),
    (re.compile("概" + _CJK_LABEL_WS + "要"), "概要"),
    (re.compile("提" + _CJK_LABEL_WS + "要"), "提要"),
    (re.compile("梗" + _CJK_LABEL_WS + "概"), "梗概"),
    (re.compile("关" + _CJK_LABEL_WS + "键" + _CJK_LABEL_WS + "词"), "关键词"),
]


def _collapse_cjk_summary_labels(s: str) -> str:
    """把被 PDF 字间距拆开的摘要/关键词标签折叠回连续写法."""
    if not s:
        return s
    for pat, repl in _CJK_SUMMARY_LABEL_COLLAPSE_RES:
        s = pat.sub(repl, s)
    return s


def _normalize_text(s: str) -> str:
    """对 PDF 解析出来的纯文本做最小集合的规范化 (CJK + ASCII 字间距).

    仅对 ``_flatten_inline`` 的最终拼接结果调用, 不要在 equation_inline 的
    LaTeX 上跑 (LaTeX 里像 ``E = m c^2`` 这种空白是结构, 不能折叠).

    返回值会 ``.strip()`` 两端: PDF 解析常把标题/段落首尾带上零宽空格或全角空格,
    这些对于摘要检测 / 标题匹配毫无用处, 反而干扰前缀比较. 各调用点之前其实
    都已 lstrip / strip, 这里冗余 strip 一下无副作用.
    """
    if not s:
        return s
    s = _normalize_spaced_cjk(s)
    s = _normalize_spaced_letters(s)
    s = _collapse_cjk_summary_labels(s)
    s = s.strip()
    return s


def _item_bbox(item: Dict[str, Any], page: int | None = None) -> Dict[str, Any]:
    raw = item.get("bbox") or item.get("box")
    if isinstance(raw, dict):
        x0 = raw.get("x0", raw.get("left", raw.get("x")))
        y0 = raw.get("y0", raw.get("top", raw.get("y")))
        x1 = raw.get("x1", raw.get("right"))
        y1 = raw.get("y1", raw.get("bottom"))
        width = raw.get("width", raw.get("w"))
        height = raw.get("height", raw.get("h"))
        if x1 is None and x0 is not None and width is not None:
            x1 = float(x0) + float(width)
        if y1 is None and y0 is not None and height is not None:
            y1 = float(y0) + float(height)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x0, y0, x1, y1 = raw[:4]
    else:
        return {}
    try:
        return {
            "page": int(item.get("_page", page if page is not None else 0) or 0),
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
        }
    except (TypeError, ValueError):
        return {}


def _union_bboxes(boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    boxes = [b for b in boxes if b and isinstance(b, dict)]
    if not boxes:
        return {}
    page = int(boxes[0].get("page", 0) or 0)
    same_page = [b for b in boxes if int(b.get("page", -1)) == page] or boxes[:1]
    return {
        "page": page,
        "x0": min(float(b["x0"]) for b in same_page),
        "y0": min(float(b["y0"]) for b in same_page),
        "x1": max(float(b["x1"]) for b in same_page),
        "y1": max(float(b["y1"]) for b in same_page),
    }


def _attach_bboxes(chunk: Dict[str, Any], boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [b for b in boxes if b]
    if clean:
        chunk["bbox"] = _union_bboxes(clean)
        chunk["bboxes"] = clean
    return chunk


def _compile_patterns(raw: Optional[List[str]]) -> List[re.Pattern[str]]:
    """把 yaml 字符串 list 编成 regex; None/空时返回空 list."""
    if not raw:
        return []
    out: List[re.Pattern[str]] = []
    for s in raw:
        try:
            out.append(re.compile(s))
        except re.error as e:
            logger.warning(f"[summary] 无效 regex {s!r}, 跳过: {e}")
    return out


# ── BM25 (零依赖, ~50 行) ────────────────────────────────────────────
# 用于 tier 2 的 section title 关键词匹配, 不依赖 Milvus / rank_bm25 第三方库.
# 适用场景: 文档集 (= 所有 section title) 数量很小 (5-30), 短文本; 不追求大库性能.
_BM25_TOKEN_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_BM25_TOKEN_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _bm25_tokenize(s: str) -> List[str]:
    """简单 tokenizer: 英文按 word 切, 中文按字切, 全部 lowercase."""
    if not s:
        return []
    s = s.lower()
    tokens: List[str] = []
    tokens.extend(_BM25_TOKEN_EN_RE.findall(s))
    tokens.extend(_BM25_TOKEN_CJK_RE.findall(s))
    return tokens


def _bm25_score_corpus(
    queries: List[str], documents: List[str], k1: float = 1.5, b: float = 0.75,
) -> List[float]:
    """给定 N 个文档, 对每个 query 算 BM25 分; 每个文档取所有 query 的 max 作为最终分.

    返回长度 = len(documents). 文档全空或 queries 全空时返回 0.0 列表.
    """
    if not documents or not queries:
        return [0.0] * len(documents)
    doc_tokens = [_bm25_tokenize(d) for d in documents]
    avgdl = sum(len(toks) for toks in doc_tokens) / max(1, len(doc_tokens))
    if avgdl <= 0:
        return [0.0] * len(documents)
    N = len(doc_tokens)
    # df: token -> 含它的文档数
    df: Dict[str, int] = {}
    for toks in doc_tokens:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1

    def _idf(token: str) -> float:
        n = df.get(token, 0)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    out: List[float] = [0.0] * N
    for q in queries:
        q_tokens = _bm25_tokenize(q)
        if not q_tokens:
            continue
        for i, toks in enumerate(doc_tokens):
            if not toks:
                continue
            dl = len(toks)
            score = 0.0
            tf: Dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w in q_tokens:
                if w not in tf:
                    continue
                f = tf[w]
                score += _idf(w) * (f * (k1 + 1.0)) / (
                    f + k1 * (1.0 - b + b * dl / avgdl)
                )
            if score > out[i]:
                out[i] = score
    return out

FIG_REF_RE = re.compile(r"\b(?:Fig(?:ure|\.)?|图)\s*([0-9IVXivx]+)", re.IGNORECASE)
TAB_REF_RE = re.compile(r"\b(?:Table|表)\s*([0-9IVXivx]+)", re.IGNORECASE)

# v5: 参考文献章节判定 (用于把 paragraph 聚合成 type=references chunk)
# 命中即视为引用列表 section, 整段不走 summary 检测, 不参与正文 paragraph_index 计数.
REFERENCES_SECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^\s*references?\s*$", re.IGNORECASE),
    re.compile(r"^\s*bibliography\s*$", re.IGNORECASE),
    re.compile(r"^\s*literature\s+cited\s*$", re.IGNORECASE),
    re.compile(r"^\s*cited\s+references?\s*$", re.IGNORECASE),
    re.compile(r"^\s*works?\s+cited\s*$", re.IGNORECASE),
    re.compile(r"^\s*参\s*考\s*文\s*献\s*$"),
    re.compile(r"^\s*引用\s*文献\s*$"),
]
# MinerU 实际产物里常见的变体: 尾冒号 / 编号前缀 / OCR 后缀噪声
_REFERENCES_SECTION_STRICT_RE = re.compile(
    r"^\s*(?:\d+[\.\、]?\s*)?"
    r"(?:参\s*考\s*文\s*献|参\s*考\s*资\s*料|引用\s*文献|"
    r"references?|bibliograph(?:y|ies)|literature\s+cited|cited\s+references?|works?\s+cited)"
    r"(?:\s*[:：])?\s*(?:[（(]\s*略\s*[）)])?\s*$",
    re.IGNORECASE,
)
_REFERENCES_SECTION_LOOSE_RE = re.compile(
    r"^\s*(?:\d+[\.\、]?\s*)?参\s*考\s*文\s*献",
    re.IGNORECASE,
)
# 参考文献条目起始: [1]/［1］/［ ］(OCR 丢号)/1. 开头
_REF_ENTRY_START_RE = re.compile(
    r"^[\[［]|^\d+[\.\)、]\s",
)
_REF_INLINE_ENTRY_SPLIT_RE = re.compile(
    r"(?=[\[［]\s*\d+\s*[\]］])",
)
# MinerU reference_list 条目编号 / 结论编号 (后者是误标 reference_list 的常见情况)
_REF_LIST_NUM_START_RE = re.compile(
    r"^[\[［]\s*\d+\s*[\]］]|^[\[［]\s*[\]］]|^\d+[\.\)、]\s",
)
_REF_CONCLUSION_START_RE = re.compile(r"^\d+[）\)]\s*")
_REF_JOURNAL_HINT_RE = re.compile(
    r"\[J\]|\[M\]|\[D\]|\[C\]|\[EB/OL\]|Journal|Proceedings|et al\.|等\.",
    re.IGNORECASE,
)
# 双语论文参考文献: 英文条目常以 "SURNAME Given-name, SURNAME Given-name" 起始
_REF_AUTHOR_LINE_START_RE = re.compile(
    r"^[A-Z][A-Za-z\-]+(?:[\s\-][A-Za-z\-]+)*,\s+[A-Z][A-Za-z\-]",
)
_REF_NOISE_RE = re.compile(
    r"^(?:[\(（].*(?:上接|下转).*(?:页)?[\)）]|"
    r"(?:上接|下转).*?(?:页)?[\)）]?|"
    r"收稿日期|作者简介|参加试验|(?:编辑|编)\s|"
    r"[\(（]?(?:作者单位|许编)[\)）]?)\s*$",
    re.IGNORECASE,
)
_REF_MISCLASS_BODY_RE = re.compile(
    r"^(?:Application of|Summary of|Key words|Abstract[:\s]|摘\s*要[：:\s])",
    re.IGNORECASE,
)
# 每个 references chunk 聚合的条目数 (默认 5; 太小会让库里塞太多稀疏 chunk,
# 太大会让 BM25 召回时单 chunk 命中多条引用难定位).
DEFAULT_REFERENCES_BATCH_SIZE = 5
# 正文短段落合并: 小于该字符数且不像新块起始 -> 与上一段合并
SHORT_PARAGRAPH_MERGE_CHARS = 40
_NEW_BLOCK_START_RE = re.compile(
    r"^(?:关键词|Key\s*words|收稿日期|参考文献|Abstract|摘\s*要|"
    r"(?:[\(（]\s*)?(?:上接|下转)|\d+[\.\)、]\s+\S)",
    re.IGNORECASE,
)
# 上一段命中以下"自闭合 listing 行"模式时, 即便它没有句末标点 / 长度短,
# 也绝不应被合并到后面的正文里. 适用场景: 期刊"关键词: A; B; C" 一行后紧跟正文.
# 注意: 这里**不包含** 摘要 / Abstract — 它们常常和正文 / 关键词同段, 应保留合并机会.
_PREV_SELF_CONTAINED_LABEL_RE = re.compile(
    r"^(?:关键词|Key\s*words?|关键字)\s*[:：]?",
    re.IGNORECASE,
)

# 句尾终止符 (用于检测 MinerU 是否误切了段落)
_SENT_TERMINATORS = "。！？!?…．"
# 段尾常见 "尾巴" 字符 (引号 / 括号 / 卷标), 计算实际句末标点时先剥掉
_TRAILING_NOISE_RE = re.compile(r"(?:\s|[\]\)\}\u201d\u2019\u300d\u300f\u3015\"'])+$")
# 段尾的引用 [1] / [1, 2] / [1-3] 等, 视为可剥离的尾巴
_TRAILING_CITATION_RE = re.compile(r"\[[0-9,\s\-–—a-zA-Z]+\]\s*$")
# 用于判断下一段是否像 "断句的延续"
_CONTINUATION_LEAD_RE = re.compile(
    r"^(?:and|or|but|nor|yet|so|that|which|while|where|when|because|though|although|"
    r"however|therefore|thus|then|moreover|furthermore|whereas|且|并|而|但|而且|"
    r"因此|因而|所以|于是|然而|然后|此外|其中|且|与)\b",
    re.IGNORECASE,
)
_NUMBERED_HEAD_RE = re.compile(r"^\d+(?:\.\d+)*\s+\S")
# LLM 兜底摘要的 prompt (无 fence)
LLM_SUMMARY_SYSTEM = (
    "你是一名严谨的科研文献编辑。给定一篇文献的第一页正文片段, "
    "请用 3-5 句话生成一段用于检索的摘要 (200-400 字), 仅使用片段中出现的事实, "
    "不要编造。直接输出摘要正文, 不要加标题, 不要加 markdown 围栏。"
)
LLM_SUMMARY_USER_TEMPLATE = (
    "以下是文献第一页的正文片段 (按出现顺序拼接):\n\n{first_page_text}\n\n"
    "请输出该文献的摘要正文。"
)
# 截断喂给 LLM 的第一页正文长度 (字符), 避免超出小模型 context
LLM_SUMMARY_MAX_INPUT_CHARS = 6000
# tier4 chat 默认调用参数 (可被 chunking.summary.llm 覆盖)
LLM_SUMMARY_DEFAULT_MAX_TOKENS = 1024
LLM_SUMMARY_DEFAULT_TEMPERATURE = 0.0
LLM_SUMMARY_DEFAULT_DISABLE_THINKING = True


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _resolve_path(root: str, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    if not root:
        return rel
    return os.path.normpath(os.path.join(root, rel))


def _flatten_inline(items: Optional[List[Dict[str, Any]]]) -> str:
    if not items:
        return ""
    parts: List[str] = []
    for it in items:
        t = it.get("type")
        c = (it.get("content") or "").strip()
        if not c:
            continue
        if t == "text":
            parts.append(c)
        elif t == "equation_inline":
            parts.append(f"${c}$")
        else:
            parts.append(c)
    joined = re.sub(r"\s+", " ", " ".join(parts)).strip()
    # PDF OCR 字间距 artifact 在 _flatten_inline 处一次性规范化, 让下游所有
    # paragraph / title / list 内容都已是干净文本; equation_inline 被 $...$ 包住,
    # 内部 LaTeX 不会被 normalize_text 触碰 (\b 不会进入 $).
    return _normalize_text(joined)


def _extract_caption_label(caption: str, kind: str) -> Optional[str]:
    if not caption:
        return None
    pat = FIG_REF_RE if kind == "image" else TAB_REF_RE
    m = pat.search(caption)
    return m.group(1).upper() if m else None


def _scan_cross_refs(text: str) -> Dict[str, List[str]]:
    figs = sorted({m.group(1).upper() for m in FIG_REF_RE.finditer(text)})
    tabs = sorted({m.group(1).upper() for m in TAB_REF_RE.finditer(text)})
    return {"figures": figs, "tables": tabs}


def is_garbled_text(s: Optional[str], min_meaningful_ratio: float = 0.3) -> bool:
    """启发式判断字符串是否为 PDF 字体映射失败造成的乱码。

    例如某些 PDF 用非标准 CMap 或 Type3 字体, MinerU 解析后 section title 会变成
    `!"#"$ &'()*+,-` 这种纯 ASCII 标点序列, 没有任何中英文/数字内容。

    判定标准:
      - 字符串非空白字符 ≤ 2: 不判乱码 (太短无法判断, 放过)
      - 否则计算 (中文 + 英文字母 + 数字) 占非空白字符总数的比例;
        低于 min_meaningful_ratio (默认 30%) 视为乱码.
    """
    if not s:
        return False
    stripped = "".join(s.split())
    if len(stripped) <= 2:
        return False
    meaningful = 0
    for ch in stripped:
        cp = ord(ch)
        # CJK Unified Ideographs (常用 + 扩展A) + 日韩等; 这里只考虑科研文献常见的中文 + ASCII
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            meaningful += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            meaningful += 1
        elif "0" <= ch <= "9":
            meaningful += 1
    ratio = meaningful / len(stripped)
    return ratio < min_meaningful_ratio


def sanitize_section(section: Optional[str]) -> str:
    """如果 section 是乱码就返回空串, 否则原样返回 (已 strip)。"""
    if not section:
        return ""
    s = section.strip()
    if not s or is_garbled_text(s):
        return ""
    return s


def _extract_title_text(item: Dict[str, Any]) -> str:
    if item.get("type") != "title":
        return ""
    raw = _flatten_inline(((item.get("content") or {}).get("title_content") or []))
    return sanitize_section(raw)


def _extract_paragraph_text(item: Dict[str, Any]) -> str:
    t = item.get("type")
    cd = item.get("content") or {}
    if t == "paragraph":
        return _flatten_inline(cd.get("paragraph_content", []))
    if t == "equation_interline":
        return (cd.get("math_content") or "").strip()
    if t == "list":
        lines: List[str] = []
        for li in cd.get("list_items") or []:
            txt = _flatten_inline(li.get("item_content", []))
            if txt:
                lines.append(txt)
        return " ".join(lines).strip()
    return ""


def _match_any(patterns: List[re.Pattern[str]], text: str) -> bool:
    return bool(text) and any(p.search(text) for p in patterns)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _collect_ordered_titles(data: List[List[Dict[str, Any]]]) -> List[str]:
    titles: List[str] = []
    for page_items in data:
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "title":
                title = _extract_title_text(item)
                if title:
                    titles.append(title)
    return titles


def _run_tier3_only(
    result: Dict[str, Any],
    normalized_titles: List[str],
    norm_to_orig: Dict[str, str],
    summary_query_texts: List[str],
    embedder: "EmbeddingClient",
    summary_sim_threshold: float,
    max_summary_sections: int,
) -> Dict[str, Any]:
    """tier 2 跳过后直接跑 tier 3 (向量相似度) 的小 helper, 避免主函数嵌套过深."""
    try:
        vectors = embedder.embed_all(summary_query_texts + normalized_titles)
    except Exception as e:
        logger.warning(
            f"[summary-detect][tier3-embedding] embedding 失败, 跳过: {e}"
        )
        return result
    if len(vectors) != len(summary_query_texts) + len(normalized_titles):
        return result
    query_vecs = vectors[: len(summary_query_texts)]
    title_vecs = vectors[len(summary_query_texts):]
    emb_scored: List[tuple] = []
    for n, tvec in zip(normalized_titles, title_vecs):
        combined = sum(_cosine_similarity(qvec, tvec) for qvec in query_vecs)
        emb_scored.append((n, combined))
    emb_scored.sort(key=lambda kv: -kv[1])
    result["embedding_top"] = [
        (norm_to_orig.get(n, n), round(s, 4))
        for n, s in emb_scored[:max_summary_sections]
    ]
    emb_picked: set = set()
    for n, s in emb_scored:
        if s >= summary_sim_threshold and len(emb_picked) < max_summary_sections:
            emb_picked.add(norm_to_orig.get(n, n))
    if emb_picked:
        logger.info(
            f"[summary-detect][tier3-embedding] threshold={summary_sim_threshold} 命中: "
            f"{[(norm_to_orig.get(n, n), round(s, 3)) for n, s in emb_scored if norm_to_orig.get(n, n) in emb_picked]}"
        )
        result["summary_sections"] = emb_picked
        result["strategy"] = "embedding"
    else:
        logger.info(
            f"[summary-detect][tier3-embedding] threshold={summary_sim_threshold} 未命中; "
            f"top: {result['embedding_top']}"
        )
    return result


def _detect_summary_sections(
    data: List[List[Dict[str, Any]]],
    summary_sim_threshold: float = SUMMARY_SIM_THRESHOLD,
    embedder: Optional[EmbeddingClient] = None,
    summary_query_texts: Optional[List[str]] = None,
    max_summary_sections: int = 2,
    # ── v6: tier 化的 summary 检测 (LLM 兜底放在调用方, 这里只到 tier 3) ──
    title_patterns: Optional[List[re.Pattern[str]]] = None,
    text_patterns: Optional[List[re.Pattern[str]]] = None,
    bm25_query_texts: Optional[List[str]] = None,
    bm25_threshold: float = SUMMARY_BM25_THRESHOLD,
    bm25_enabled: bool = True,
    embedding_enabled: bool = True,
) -> Dict[str, Any]:
    """v6: 4 级 fallback 的强信号摘要检测; LLM 兜底放在 ``build_knowledge_blocks``.

    优先级 (由强到弱, 命中即返回, 不再触发下一级):

    - **tier 1 — 字面强匹配 (rule)**:
        section title 命中 ``title_patterns`` 任一 regex, 或前 2 页正文段开头
        命中 ``text_patterns`` 任一 regex (后者只设 text_hit, 让 _flush_section
        在每个 section 内做 "段首核验"). 命中即定, 无阈值.

    - **tier 2 — BM25 关键词匹配 (bm25)**:
        把所有 section title 当文档库, 对 ``bm25_query_texts`` 每条 query 算
        BM25 分, 每个 title 取所有 query 的 max 作为最终得分. ≥ bm25_threshold
        视为强信号.

    - **tier 3 — 向量相似度 (embedding)**:
        section title 与 ``summary_query_texts`` (三组词包) 的 dense 余弦相似度
        之和 ≥ summary_sim_threshold 视为强信号.

    - **tier 4 — LLM 兜底**:
        本函数不负责; 由调用方 (build_knowledge_blocks) 在 strategy=="none" 时调
        ``_llm_synthesize_summary`` 注入一个新的 summary chunk.

    text_hit 是 tier 1 的副产物, 单独保留: 即使 title 没命中任何 tier, 只要前 2 页
    正文段首出现 "Abstract:" / "摘要:" 这样的内联信号, _flush_section 仍能在该 section
    第一段把 summary 拆出来.

    Returns:
        dict with: summary_sections, strategy (rule/bm25/embedding/none),
        bm25_top, embedding_top, ordered_titles, text_hit
    """
    summary_query_texts = summary_query_texts or SUMMARY_QUERY_TEXTS
    title_patterns = title_patterns or SUMMARY_TITLE_PATTERNS
    text_patterns = text_patterns or SUMMARY_TEXT_PATTERNS
    bm25_query_texts = bm25_query_texts or SUMMARY_BM25_QUERY_TEXTS

    ordered_titles = _collect_ordered_titles(data)
    # 关键: 给每个 title 做一次 spaced-letters 归一化, 让 "A B S T R A C T"
    # 这种 PDF 字间距 artifact 能匹配上 \babstract\b / BM25 token "abstract".
    # 匹配命中后, 返回的仍是原始 title (保持下游 cur_section 比对).
    normalized_titles = [_normalize_text(t) for t in ordered_titles]
    norm_to_orig: Dict[str, str] = {
        n: t for n, t in zip(normalized_titles, ordered_titles)
    }

    # text_hit: 前 2 页正文段开头出现 "Abstract:" / "摘要:" 等. 与 tier 系统独立,
    # _flush_section 在 section 内会用到. 这里也做 normalize 以处理 "A b s t r a c t:".
    text_hit = False
    for page_items in data[:2]:
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in ("paragraph", "equation_interline", "list"):
                continue
            para_text = _normalize_text(_extract_paragraph_text(item))
            if _match_any(text_patterns, para_text):
                text_hit = True
                break
        if text_hit:
            break

    result: Dict[str, Any] = {
        "summary_sections": set(),
        "strategy": "none",
        "bm25_top": [],
        "embedding_top": [],
        "ordered_titles": ordered_titles,
        "text_hit": text_hit,
    }

    if not ordered_titles:
        return result

    # ── tier 1: 字面强匹配 (在 normalized title 上跑) ──
    rule_hits_norm = {
        n for n in normalized_titles if _match_any(title_patterns, n)
    }
    if rule_hits_norm:
        kept: set = set()
        for n in normalized_titles:
            if n in rule_hits_norm and len(kept) < max_summary_sections:
                kept.add(norm_to_orig.get(n, n))
        logger.info(
            f"[summary-detect][tier1-rule] 命中 {len(kept)} 个 title (≤{max_summary_sections}): "
            f"{sorted(kept)}"
        )
        result["summary_sections"] = kept
        result["strategy"] = "rule"
        return result

    # ── tier 2: BM25 关键词匹配 (在 normalized title 上跑) ──
    if not bm25_enabled:
        logger.info("[summary-detect][tier2-bm25] 跳过: bm25.enabled=false")
        # 跳过 tier 2, 直接走 tier 3
        if not embedding_enabled or embedder is None:
            logger.info("[summary-detect][tier3-embedding] 跳过: embedding.enabled=false 或未提供 embedder")
            return result
        # 走 tier 3
        return _run_tier3_only(
            result, normalized_titles, norm_to_orig, summary_query_texts,
            embedder, summary_sim_threshold, max_summary_sections,
        )
    bm25_scores = _bm25_score_corpus(bm25_query_texts, normalized_titles)
    bm25_scored = sorted(
        zip(normalized_titles, bm25_scores), key=lambda kv: -kv[1],
    )
    result["bm25_top"] = [
        (norm_to_orig.get(n, n), round(s, 4))
        for n, s in bm25_scored[:max_summary_sections]
    ]
    bm25_picked: set = set()
    for n, s in bm25_scored:
        if s >= bm25_threshold and len(bm25_picked) < max_summary_sections:
            bm25_picked.add(norm_to_orig.get(n, n))
    if bm25_picked:
        logger.info(
            f"[summary-detect][tier2-bm25] threshold={bm25_threshold} 命中: "
            f"{[(norm_to_orig.get(n, n), round(s, 3)) for n, s in bm25_scored if norm_to_orig.get(n, n) in bm25_picked]}"
        )
        result["summary_sections"] = bm25_picked
        result["strategy"] = "bm25"
        return result
    logger.info(
        f"[summary-detect][tier2-bm25] threshold={bm25_threshold} 未命中; "
        f"top: {result['bm25_top']}"
    )

    # ── tier 3: 向量相似度 (用 normalized title 编码; 比 spaced 版本语义更稳) ──
    if not embedding_enabled:
        logger.info("[summary-detect][tier3-embedding] 跳过: embedding.enabled=false")
        return result
    if embedder is None:
        logger.info("[summary-detect][tier3-embedding] 跳过: 未提供 embedder")
        return result
    try:
        vectors = embedder.embed_all(summary_query_texts + normalized_titles)
    except Exception as e:
        logger.warning(
            f"[summary-detect][tier3-embedding] embedding 失败, 跳过: {e}"
        )
        return result
    if len(vectors) != len(summary_query_texts) + len(normalized_titles):
        logger.warning(
            "[summary-detect][tier3-embedding] embedding 返回数量不匹配, 跳过"
        )
        return result
    query_vecs = vectors[: len(summary_query_texts)]
    title_vecs = vectors[len(summary_query_texts):]
    emb_scored: List[tuple] = []
    for n, tvec in zip(normalized_titles, title_vecs):
        combined = sum(_cosine_similarity(qvec, tvec) for qvec in query_vecs)
        emb_scored.append((n, combined))
    emb_scored.sort(key=lambda kv: -kv[1])
    result["embedding_top"] = [
        (norm_to_orig.get(n, n), round(s, 4))
        for n, s in emb_scored[:max_summary_sections]
    ]
    emb_picked: set = set()
    for n, s in emb_scored:
        if s >= summary_sim_threshold and len(emb_picked) < max_summary_sections:
            emb_picked.add(norm_to_orig.get(n, n))
    if emb_picked:
        logger.info(
            f"[summary-detect][tier3-embedding] threshold={summary_sim_threshold} 命中: "
            f"{[(norm_to_orig.get(n, n), round(s, 3)) for n, s in emb_scored if norm_to_orig.get(n, n) in emb_picked]}"
        )
        result["summary_sections"] = emb_picked
        result["strategy"] = "embedding"
        return result
    logger.info(
        f"[summary-detect][tier3-embedding] threshold={summary_sim_threshold} 未命中; "
        f"top: {result['embedding_top']}"
    )
    # tier 1-3 都没命中 → strategy=none, 等 tier 4 (LLM) 在 build_knowledge_blocks 兜底
    return result


def _strip_summary_prefix(
    text: str, text_patterns: Optional[List[re.Pattern[str]]] = None,
) -> str:
    cleaned = text.strip()
    for pat in (text_patterns or SUMMARY_TEXT_PATTERNS):
        cleaned = pat.sub("", cleaned, count=1).strip()
    return cleaned


def _clip_summary_text(
    text: str, stop_patterns: Optional[List[re.Pattern[str]]] = None,
) -> str:
    if not text:
        return ""
    cut_pos: Optional[int] = None
    for pat in (stop_patterns or SUMMARY_STOP_PATTERNS):
        m = pat.search(text)
        if m and (cut_pos is None or m.start() < cut_pos):
            cut_pos = m.start()
    if cut_pos is None:
        return text.strip()
    return text[:cut_pos].strip()


def _match_any_at_start(
    patterns: List[re.Pattern[str]], text: str,
) -> bool:
    """是否有 pattern 严格匹配在 text 的起始位置 (用 re.match, 不是 re.search).

    比 search 更安全, 避免把段中出现 "摘要" 的 paragraph 误判为段首命中.
    """
    if not text or not patterns:
        return False
    for pat in patterns:
        if pat.match(text):
            return True
    return False


def _promote_text_to_summary(
    chunks: List[Dict[str, Any]],
    text_patterns: Optional[List[re.Pattern[str]]] = None,
    stop_patterns: Optional[List[re.Pattern[str]]] = None,
    max_promote: int = 1,
    scan_head_chars: int = 30,
    require_first_page: bool = True,
) -> int:
    """v6: post-process 段首关键词提升 (content-driven summary promotion).

    关键场景: PDF 里没有 "Abstract" / "摘要" section title, 或者 section title 被
    解析失败成 "" / 噪音, 但 content 第一句就是 "摘要：本文主要研究了..." /
    "Abstract: This paper presents...". section-title 级别的 tier 1-3 都 miss,
    但用户期望这种段落被识别为摘要 (这是用户反馈中报的真实 case).

    规则:
    - 只扫 ``type == "text"`` 的 chunk; 已经是 summary 的不再覆盖
    - 只看 content 的前 ``scan_head_chars`` 字符是否命中 ``text_patterns``
    - ``require_first_page=True`` 时, chunk 必须在 page 0 (pages[0]==0), 避免误
      命中 "summary of results" 这种出现在结论段的字眼
    - 命中后: type→summary; content 剥前缀 + clip 尾部 (用同一套 stop_patterns)
    - 命中数受 ``max_promote`` 约束 (与已存在 summary 计入同一个上限),
      避免一篇文献产生多个 "假 summary"

    Returns:
        promoted 数量 (0 表示没有 chunk 被提升)
    """
    text_patterns = text_patterns or SUMMARY_TEXT_PATTERNS
    if not text_patterns:
        return 0
    existing_summary = sum(1 for c in chunks if c.get("type") == "summary")
    if existing_summary >= max_promote:
        return 0

    promoted = 0
    for c in chunks:
        if existing_summary + promoted >= max_promote:
            break
        if c.get("type") != "text":
            continue
        content = (c.get("content") or "").strip()
        if not content:
            continue
        # 严格段首匹配: 先 strip 前导空白, 取前 N 字符做 spaced-letters 归一化,
        # 再用 re.match (从 0 位严格匹配) 而不是 re.search. 这样 "段中出现摘要"
        # 不会被误判. scan_head_chars 默认 30 足够覆盖 "Abstract: ..." 这种.
        head = _normalize_text(content.lstrip()[:scan_head_chars])
        if not _match_any_at_start(text_patterns, head):
            continue
        if require_first_page:
            pages = c.get("pages") or []
            if not pages or min(pages) != 0:
                continue
        cleaned = _strip_summary_prefix(content, text_patterns=text_patterns)
        cleaned = _clip_summary_text(cleaned, stop_patterns=stop_patterns)
        if not cleaned:
            continue
        old_section = c.get("section") or ""
        c["type"] = "summary"
        c["content"] = cleaned
        # 标记: 这是 content-driven 提升, 不是 tier4 LLM 合成
        c["synthesized"] = False
        promoted += 1
        logger.info(
            f"[summary-promote] text chunk {c.get('id')} 段首命中 → 提升为 summary "
            f"(section={old_section!r}, content_len={len(cleaned)})"
        )
    return promoted


def _count_sentences(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"[。！？!?；;\n]+", text)
    return sum(1 for p in parts if p.strip())


# ── 句子边界 (用于给 equation chunk 注入上下文 anchor 句) ─────────────────
# 注意: 只按真正的句末标点切 (。！？!?), 不按 ;／； 切. 因为科研文献里 "式中: ...;
# Δm 为...; ρ 为..." 整句是一个解释段, 用分号切会把它拆得没法识别 "腐蚀减薄量"
# 这种关键变量名.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[。！？!?])\s*")


def _split_sentences(text: str) -> List[str]:
    """简单按中英文句末标点切句, 用于 anchor / context 注入. 不追求 NLP 精度."""
    if not text:
        return []
    parts = _SENT_BOUNDARY_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_first_sentences(text: str, n: int = 1) -> str:
    sents = _split_sentences(text)
    return " ".join(sents[: max(1, n)]).strip()


def _extract_last_sentences(text: str, n: int = 1) -> str:
    sents = _split_sentences(text)
    return " ".join(sents[-max(1, n):]).strip()


# ---------------------------------------------------------------------------
# 段落合并启发式: 判断 MinerU 是否把一个段落拆成两段
# ---------------------------------------------------------------------------

def _strip_trailing_noise(text: str) -> str:
    """剥掉段尾的引号/括号/卷标/引用 [1] 等噪声后再看真正的末尾字符。"""
    if not text:
        return ""
    s = text.rstrip()
    # 反复剥离: 引用 + 括号引号尾巴
    for _ in range(3):
        s2 = _TRAILING_CITATION_RE.sub("", s).rstrip()
        s2 = _TRAILING_NOISE_RE.sub("", s2).rstrip()
        if s2 == s:
            break
        s = s2
    return s


def _ends_with_terminator(text: str) -> bool:
    """段尾是否带有完整句子终止符。"""
    s = _strip_trailing_noise(text)
    if not s:
        return False
    return s[-1] in _SENT_TERMINATORS or s[-1] == "."


def _looks_like_continuation(text: str) -> bool:
    """下一段开头是否像 "上一段的延续"。

    判定依据:
    1. 首字符是英文小写字母 -> 几乎一定是断句
    2. 首字符是中文 + 上一段无终止符 -> 通常也是断句 (中文学术文常见)
    3. 以连接词 / 关系词起始 -> 断句
    4. 以 1./1.1 等编号 + 文字开头 -> 这是新章节/新段, 不算延续
    """
    if not text:
        return True
    c = text[0]
    if _NUMBERED_HEAD_RE.match(text):
        return False
    if c.islower() and c.isascii():
        return True
    if _CONTINUATION_LEAD_RE.match(text):
        return True
    return False


def _should_merge_paragraphs(prev_text: str, cur_text: str) -> bool:
    """判断两段相邻 paragraph 是否应该合并为一个 "逻辑段落"。

    规则:
    - 前段是 "关键词: ..." 这种自闭合 listing 行 -> 一律不合并
    - 前段无句末终止符 + (后段以小写英文起始 OR 后段以连接词起始 OR 后段不是数字编号开头)
    - 前段以冒号结尾 (小节标签行) + 后段不像新块起始
    - 后段极短 (< SHORT_PARAGRAPH_MERGE_CHARS) + 前段无终止符 + 后段不像新块起始
    - 任意一段为空 -> 不合并
    """
    if not prev_text or not cur_text:
        return False
    cur_stripped = cur_text.strip()
    prev_stripped = prev_text.strip()
    # 上一段是 "关键词: A; B; C" 这种自闭合 listing 行, 后面紧跟的正文不能粘进来,
    # 否则得到的 chunk 头是关键词列表尾巴 + 正文首句, 语义错位 (BM25 / dense 都受影响).
    if _PREV_SELF_CONTAINED_LABEL_RE.match(prev_stripped):
        return False
    if _NEW_BLOCK_START_RE.match(cur_stripped):
        return False
    if prev_text.rstrip().endswith(("：", ":")):
        return True
    if (
        len(cur_stripped) < SHORT_PARAGRAPH_MERGE_CHARS
        and not _ends_with_terminator(prev_text)
        and not _NUMBERED_HEAD_RE.match(cur_stripped)
    ):
        return True
    if _ends_with_terminator(prev_text):
        return False
    if _NUMBERED_HEAD_RE.match(cur_text):
        return False
    return _looks_like_continuation(cur_text) or not cur_text[0].isupper()


def _join_paragraph_fragments(fragments: List[str]) -> str:
    """把被误切的 paragraph 片段拼成一段; 中文不加空格, 英文加空格。"""
    if not fragments:
        return ""
    out = fragments[0].rstrip()
    for frag in fragments[1:]:
        frag = frag.lstrip()
        if not frag:
            continue
        if not out:
            out = frag
            continue
        last_c = out[-1]
        first_c = frag[0]
        if "\u4e00" <= last_c <= "\u9fff" or "\u4e00" <= first_c <= "\u9fff":
            out += frag
        else:
            out += " " + frag
    return out.strip()


# ---------------------------------------------------------------------------
# chunk 构建函数 (段落级)
# ---------------------------------------------------------------------------

def _group_logical_paragraphs(
    items: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """把一个 section 内的 items 划分为若干 "逻辑段落组"。

    规则:
    - paragraph 之间走 _should_merge_paragraphs 启发式合并
    - equation_interline / list 单独成组 (不与文本合并, 但保留就近顺序)
    - paragraph 后紧跟 equation_interline + 又是 paragraph 时, equation
      会作为独立组插在中间, 不破坏前后段的合并判断
    """
    groups: List[List[Dict[str, Any]]] = []
    last_para_text = ""
    for item in items:
        t = item.get("type")
        if t == "equation_interline" or t == "list":
            groups.append([item])
            last_para_text = ""
            continue
        if t != "paragraph":
            continue
        cur_text = _extract_paragraph_text(item)
        if not cur_text:
            continue
        # 期刊元数据行 (中图分类号 / 文献标识码 / 文章编号 / DOI / 收稿日期 等) 直接丢弃,
        # 不进 chunk, 也不影响 last_para_text 的合并判断 (上一段不会被 metadata 行"封口").
        if _is_metadata_line(cur_text):
            logger.debug(f"[chunker] drop metadata line: {cur_text!r}")
            continue
        if (
            groups
            and groups[-1]
            and groups[-1][-1].get("type") == "paragraph"
            and _should_merge_paragraphs(last_para_text, cur_text)
        ):
            groups[-1].append(item)
            last_para_text = _join_paragraph_fragments(
                [last_para_text, cur_text]
            )
        else:
            groups.append([item])
            last_para_text = cur_text
    return _merge_short_paragraph_groups(groups)


def _group_combined_text(group: List[Dict[str, Any]]) -> str:
    """把一个逻辑段落组的纯文本拼起来 (仅 paragraph)。"""
    parts = [
        _extract_paragraph_text(p)
        for p in group
        if p.get("type") == "paragraph"
    ]
    return _join_paragraph_fragments([p for p in parts if p])


def _merge_short_paragraph_groups(
    groups: List[List[Dict[str, Any]]],
    min_chars: int = SHORT_PARAGRAPH_MERGE_CHARS,
) -> List[List[Dict[str, Any]]]:
    """把过短的 paragraph-only 组与相邻组合并, 减少碎片 chunk。"""
    if len(groups) <= 1:
        return groups
    out: List[List[Dict[str, Any]]] = []
    for group in groups:
        if not group:
            continue
        para_only = all(p.get("type") == "paragraph" for p in group)
        cur_text = _group_combined_text(group) if para_only else ""
        if (
            out
            and para_only
            and all(p.get("type") == "paragraph" for p in out[-1])
            and cur_text
            and len(cur_text) < min_chars
        ):
            prev_text = _group_combined_text(out[-1])
            if (
                prev_text
                and not _NEW_BLOCK_START_RE.match(cur_text.strip())
                and (
                    not _ends_with_terminator(prev_text)
                    or len(prev_text) < min_chars
                )
            ):
                out[-1].extend(group)
                continue
        out.append(list(group))
    return out


def _build_chunk_from_group(
    group: List[Dict[str, Any]],
    section: str,
    pages: List[int],
    is_summary: bool,
    paragraph_index: int,
    summary_text_patterns: Optional[List[re.Pattern[str]]] = None,
    summary_stop_patterns: Optional[List[re.Pattern[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """从一个 "逻辑段落组" 构建单个 chunk。

    group 内可能是:
    - 多个 paragraph (它们被启发式判定为同一逻辑段, 拼接)
    - 单个 equation_interline (独立 chunk, type=text)
    - 单个 list (合成多行 bullet, type=text)
    """
    fragments: List[str] = []
    equations: List[str] = []
    only_equation = (
        len(group) == 1 and group[0].get("type") == "equation_interline"
    )
    for p in group:
        t = p.get("type")
        cd = p.get("content") or {}
        if t == "paragraph":
            txt = _flatten_inline(cd.get("paragraph_content", []))
            if txt:
                fragments.append(txt)
        elif t == "equation_interline":
            math_content = (cd.get("math_content") or "").strip()
            if math_content:
                snippet = (
                    math_content if math_content.startswith("$$")
                    else f"$$\n{math_content}\n$$"
                )
                if only_equation:
                    fragments.append(snippet)
                else:
                    equations.append(math_content)
        elif t == "list":
            list_lines: List[str] = []
            for li in cd.get("list_items") or []:
                txt = _flatten_inline(li.get("item_content", []))
                if txt:
                    list_lines.append(f"- {txt}")
            if list_lines:
                fragments.append("\n".join(list_lines))

    if len(group) > 1 and group[0].get("type") == "paragraph":
        # 多个 paragraph 片段 -> 启发式合并为一段连续文字
        para_fragments = [
            _flatten_inline((p.get("content") or {}).get("paragraph_content", []))
            for p in group if p.get("type") == "paragraph"
        ]
        merged = _join_paragraph_fragments([f for f in para_fragments if f])
        if merged:
            fragments = [merged]

    content = "\n\n".join(p for p in fragments if p.strip()).strip()
    if not content:
        return None

    if is_summary:
        content = _clip_summary_text(
            _strip_summary_prefix(content, text_patterns=summary_text_patterns),
            stop_patterns=summary_stop_patterns,
        )
        if not content:
            return None

    # v5: 单一 equation_interline 提升为独立 equation 类 (而非塞进 text 的 context),
    # 方便检索端按 chunk_type=equation 专项召回, 同时由后处理 _link_equations_to_text
    # 在该 chunk 和前后 text/summary chunk 之间建立双向 related_assets.
    # is_summary 优先级最高 (摘要 section 内的公式仍并入摘要文字).
    if is_summary:
        chunk_type = "summary"
    elif only_equation:
        chunk_type = "equation"
    else:
        chunk_type = "text"
    chunk = {
        "id": _short_id(chunk_type),
        "type": chunk_type,
        "section": section,
        "pages": sorted(set(pages)),
        "content": content,
        "context": "\n\n".join(equations) if equations else "",
        "related_assets": [],
        # equation 是 "asset-like" (非正文段落), paragraph_index 标为 -1, 与 image/table 对齐;
        # 这样 local 路径按 paragraph_index in [...] 召回时不会拿到公式,
        # 用户问 "第 12 段" 仍只命中真正的 text/summary.
        "paragraph_index": -1 if chunk_type == "equation" else paragraph_index,
    }
    return _attach_bboxes(chunk, [_item_bbox(item) for item in group])


# 兼容外部可能的调用 (现保留, 内部不再使用; section 模式仍可一键合成全章 chunk)
def _build_text_or_summary_chunk(
    paragraphs: List[Dict[str, Any]],
    section: str,
    pages: List[int],
    is_summary: bool,
) -> Optional[Dict[str, Any]]:
    """整 section 合成单 chunk (legacy, 仅用于摘要检测的探测路径)。"""
    return _build_chunk_from_group(
        paragraphs, section=section, pages=pages,
        is_summary=is_summary, paragraph_index=-1,
    )


def _maybe_split_text_chunk(
    base_chunk: Dict[str, Any],
    embedder: Optional[EmbeddingClient],
    target_chars: int,
    max_chars: int,
    min_chars: int,
    breakpoint_percentile: int,
    length_fn: Optional[Callable[[str], int]] = None,
    overlap_chars: int = 0,
) -> List[Dict[str, Any]]:
    """对超长 text chunk 做语义切分; summary 不切。

    输出多个子 chunk:
    - 共享同一 parent_chunk_id, section, pages, context (公式只放在第一个子 chunk 上)
    - 子 chunk id 后缀 _p001 / _p002 ...
    - 子 chunk type 与父 chunk 一致 (text)
    """
    content = base_chunk.get("content") or ""
    _len = length_fn if length_fn is not None else len
    if base_chunk.get("type") != "text" or _len(content) <= max_chars:
        return [base_chunk]

    pieces = semantic_split(
        content, embedder,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
        breakpoint_percentile=breakpoint_percentile,
        overlap_chars=overlap_chars,
        length_fn=length_fn,
    )
    if not pieces or len(pieces) == 1:
        return [base_chunk]

    parent_id = base_chunk["id"]
    out: List[Dict[str, Any]] = []
    for idx, piece in enumerate(pieces, 1):
        sub_id = f"{parent_id}_p{idx:03d}"
        sub = dict(base_chunk)
        sub["id"] = sub_id
        sub["content"] = piece.text
        sub["parent_chunk_id"] = parent_id
        sub["chunk_index"] = idx
        sub["chunk_total"] = len(pieces)
        # paragraph_index 保留 (子 chunk 共享同一段号), 由调用方保证已写入
        # context (公式块) 只挂在第一个子 chunk 上, 避免重复
        if idx > 1:
            sub["context"] = ""
        out.append(sub)
    logger.debug(
        f"[split] section={base_chunk.get('section')!r} "
        f"len={len(content)} -> {len(pieces)} pieces"
    )
    return out


def _is_mineru_reference_list(item: Dict[str, Any]) -> bool:
    """MinerU v2 用 ``list`` + ``content.list_type == reference_list`` 标记参考文献。"""
    if item.get("type") != "list":
        return False
    cd = item.get("content") or {}
    return (cd.get("list_type") or "").strip().lower() == "reference_list"


def _reference_list_item_texts(item: Dict[str, Any]) -> List[str]:
    """取出 reference_list 每个 list_item 的纯文本 (含 equation_inline)。"""
    cd = item.get("content") or {}
    out: List[str] = []
    for li in cd.get("list_items") or []:
        txt = _flatten_inline(li.get("item_content", []))
        if txt.strip():
            out.append(txt.strip())
    return out


def _is_bibliography_reference_list(texts: List[str]) -> bool:
    """区分真正的参考文献 list 与 MinerU 误标为 reference_list 的结论条目 (1）2）… )。"""
    if not texts:
        return False
    ref_score = 0
    concl_score = 0
    for t in texts:
        s = t.strip()
        if _REF_CONCLUSION_START_RE.match(s):
            concl_score += 1
        if _REF_LIST_NUM_START_RE.match(s) or _REF_JOURNAL_HINT_RE.search(s):
            ref_score += 1
    if concl_score >= 2 and concl_score >= ref_score:
        return False
    if ref_score >= 1:
        return True
    # 跨页续行块: 无编号但含期刊特征或长英文条目
    return any(
        _REF_JOURNAL_HINT_RE.search(t)
        or (len(t.strip()) > 40 and re.search(r"[A-Za-z]{4,}", t))
        for t in texts
    )


def _normalize_section_title(section: str) -> str:
    """压缩 section 标题空白, 便于参考文献章节匹配。"""
    return re.sub(r"\s+", " ", (section or "").strip())


def _is_references_section(section: str) -> bool:
    """判断 section title 是否为参考文献章节。"""
    sec = _normalize_section_title(section)
    if not sec:
        return False
    if _REFERENCES_SECTION_STRICT_RE.match(sec):
        return True
    if _REFERENCES_SECTION_LOOSE_RE.match(sec):
        return True
    return any(p.match(sec) for p in REFERENCES_SECTION_PATTERNS)


def _is_reference_noise(text: str) -> bool:
    """过滤参考文献区里的页眉页脚/续页/摘要误识别等噪声。"""
    t = (text or "").strip()
    if not t or len(t) <= 3:
        return True
    if _REF_NOISE_RE.match(t):
        return True
    if _REF_MISCLASS_BODY_RE.match(t):
        return True
    if len(t) > 350 and re.search(r"\bAbstract\b|摘\s*要", t, re.IGNORECASE):
        return True
    return False


def _is_reference_entry_start(text: str) -> bool:
    """判断文本是否像一条新参考文献的起始。"""
    t = (text or "").strip()
    if not t:
        return False
    if _REF_ENTRY_START_RE.match(t):
        return True
    if re.match(r"^[\[［]\s*[\]］]", t):
        return True
    if _REF_AUTHOR_LINE_START_RE.match(t):
        return True
    return False


def _split_inline_reference_entries(text: str) -> List[str]:
    """把同一段内多条 [n] 引文拆成独立条目。"""
    t = (text or "").strip()
    if not t:
        return []
    parts = _REF_INLINE_ENTRY_SPLIT_RE.split(t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [t]


def _append_reference_fragment(entries: List[str], piece: str) -> None:
    """把一条引用片段追加到 entries, 自动处理跨 list item 的续行。"""
    piece = piece.strip()
    if not piece or _is_reference_noise(piece):
        return
    if _is_reference_entry_start(piece) or not entries:
        entries.append(piece)
        return
    prev = entries[-1]
    if prev.endswith(("-", "－", "—")):
        entries[-1] = prev + piece.lstrip()
    elif "\u4e00" <= prev[-1] <= "\u9fff" or "\u4e00" <= piece[0] <= "\u9fff":
        entries[-1] = prev + piece
    else:
        entries[-1] = prev + " " + piece


def _extract_entries_from_reference_list(item: Dict[str, Any]) -> List[str]:
    """从 MinerU ``reference_list`` 块提取引用条目 (含跨 item 续行合并)。"""
    entries: List[str] = []
    for txt in _reference_list_item_texts(item):
        for piece in _split_inline_reference_entries(txt):
            _append_reference_fragment(entries, piece)
    return [
        e for e in entries
        if len(e.strip()) >= 6 or _is_reference_entry_start(e)
    ]


def _extract_reference_entries(items: List[Dict[str, Any]]) -> List[str]:
    """从段落 / 普通 list 中提取引用 (section 标题兜底路径)。"""
    raw: List[str] = []
    for p in items:
        t = p.get("type")
        cd = p.get("content") or {}
        if t == "paragraph":
            txt = _flatten_inline(cd.get("paragraph_content", []))
            if txt.strip():
                raw.append(txt.strip())
        elif t == "list" and not _is_mineru_reference_list(p):
            for li in cd.get("list_items") or []:
                txt = _flatten_inline(li.get("item_content", []))
                if txt.strip():
                    raw.append(txt.strip())

    entries: List[str] = []
    for txt in raw:
        for piece in _split_inline_reference_entries(txt):
            _append_reference_fragment(entries, piece)

    return [
        e for e in entries
        if len(e.strip()) >= 6 or _is_reference_entry_start(e)
    ]


def _append_pending_reference_entries(
    pending: List[str], new_entries: List[str],
) -> None:
    """把新提取的引用条目并入 pending (跨 reference_list 块续行合并)。"""
    for entry in new_entries:
        _append_reference_fragment(pending, entry)


def _build_references_chunks_from_entries(
    entries: List[str],
    section: str,
    pages: List[int],
    batch_size: int = DEFAULT_REFERENCES_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """把已提取的引用条目 batch 成 type=references chunk。"""
    if not entries:
        return []

    out: List[Dict[str, Any]] = []
    sorted_pages = sorted(set(pages))
    section_label = _normalize_section_title(section) or "References"
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        content = "\n\n".join(batch).strip()
        if not content:
            continue
        out.append({
            "id": _short_id("references"),
            "type": "references",
            "section": section_label,
            "pages": sorted_pages,
            "content": content,
            "context": "",
            "related_assets": [],
            "paragraph_index": -1,
            "ref_index_start": i + 1,
            "ref_index_end": i + len(batch),
            "ref_count": len(batch),
        })
    return out


def _build_references_chunks(
    paragraphs: List[Dict[str, Any]],
    section: str,
    pages: List[int],
    batch_size: int = DEFAULT_REFERENCES_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """把参考文献 section 内 paragraph/list 聚合成 type=references chunk (兜底路径)。"""
    entries = _extract_reference_entries(paragraphs)
    return _build_references_chunks_from_entries(
        entries, section=section, pages=pages, batch_size=batch_size,
    )


def _link_equations_to_text(chunks: List[Dict[str, Any]]) -> None:
    """v5 后处理: 在 equation chunk 与其前/后最近的 text/summary chunk 之间建立
    双向 related_assets 链接。

    规则:
    - 在文档出现顺序里 (即 chunks 的下标顺序), 对每个 equation chunk 找:
        prev = 同 section 内下标 < eq_idx 的最后一个 text/summary chunk
        next = 同 section 内下标 > eq_idx 的第一个 text/summary chunk
    - prev/next 都加上 {type:"equation", label:"Eq. (page X, sect Y)", chunk_id:eq.id}
    - equation chunk 自身加上 {type:"text", label:"para N (page X)", chunk_id:t.id}
    - 链接是 *追加* 而不是覆盖, 避免冲掉已有的 fig/table cross-ref.
    """
    text_like = ("text", "summary")
    for eq_idx, eq in enumerate(chunks):
        if eq.get("type") != "equation":
            continue
        eq_section = eq.get("section") or ""
        eq_pages = eq.get("pages") or []
        eq_page = (min(eq_pages) + 1) if eq_pages else None
        eq_label = (
            f"Eq. (page {eq_page})" if eq_page is not None else "Equation"
        )

        prev: Optional[Dict[str, Any]] = None
        nxt: Optional[Dict[str, Any]] = None
        # 向前找最近同 section 的 text/summary
        for j in range(eq_idx - 1, -1, -1):
            c = chunks[j]
            if c.get("type") in text_like and (c.get("section") or "") == eq_section:
                prev = c
                break
        # 向后找最近同 section 的 text/summary
        for j in range(eq_idx + 1, len(chunks)):
            c = chunks[j]
            if c.get("type") in text_like and (c.get("section") or "") == eq_section:
                nxt = c
                break

        for neighbor in (prev, nxt):
            if neighbor is None or neighbor.get("id") == eq.get("id"):
                continue
            # 邻居 -> 公式
            n_assets = neighbor.setdefault("related_assets", [])
            if not any(a.get("chunk_id") == eq["id"] for a in n_assets):
                n_assets.append({
                    "type": "equation",
                    "label": eq_label,
                    "chunk_id": eq["id"],
                })
            # 公式 -> 邻居
            eq_assets = eq.setdefault("related_assets", [])
            if not any(a.get("chunk_id") == neighbor["id"] for a in eq_assets):
                para_idx = neighbor.get("paragraph_index")
                n_page = (
                    (min(neighbor.get("pages") or [0]) + 1)
                    if neighbor.get("pages") else None
                )
                if isinstance(para_idx, int) and para_idx >= 1:
                    n_label = f"para {para_idx}"
                elif neighbor.get("type") == "summary":
                    n_label = "summary"
                else:
                    n_label = neighbor.get("section") or "context"
                if n_page is not None:
                    n_label = f"{n_label} (page {n_page})"
                eq_assets.append({
                    "type": neighbor["type"],
                    "label": n_label,
                    "chunk_id": neighbor["id"],
                })

        # P1-6: 给 equation chunk 注入 context anchor 句 (BM25/dense 才能命中公式).
        # 纯 LaTeX 在向量空间几乎是噪音, 用 prev 段末 1 句 + next 段首 1 句作锚点;
        # next 段常以 "式中：" / "where" 开头, 直接解释变量, 信号最强, 放前面.
        ctx_parts: List[str] = []
        if nxt and nxt.get("content"):
            head = _extract_first_sentences(nxt["content"], n=1)
            if head:
                ctx_parts.append(head)
        if prev and prev.get("content"):
            tail = _extract_last_sentences(prev["content"], n=1)
            if tail and tail not in ctx_parts:
                ctx_parts.append(tail)
        if ctx_parts:
            # 不覆盖已有 context (理论上 equation chunk 的 context 在构造时是 ""),
            # 但保留兜底; 之间用换行分隔便于阅读.
            existing = (eq.get("context") or "").strip()
            new_ctx = "\n".join(ctx_parts)
            eq["context"] = (
                f"{existing}\n{new_ctx}".strip() if existing else new_ctx
            )


def _build_asset_chunk(
    item: Dict[str, Any], page_idx: int, section: str, images_root: str,
) -> Optional[Dict[str, Any]]:
    """从 image / table 块构建 asset chunk.

    路径策略 (P0-3, 2026-05 优化):
    - ``content`` 字段中的 ``[Image Path] xxx`` / ``[Table Image Path] xxx`` 使用
      **相对路径** (即 MinerU 原样给的 ``images/<hash>.jpg``), 不再嵌入绝对路径.
      这样:
        * 灌入 Milvus 的 ``content`` 字段不会暴露宿主机绝对路径给 LLM;
        * BM25 倒排里 token 只有 ``images`` + 文件 hash, 无跨文档串扰;
        * knowledge_blocks.json 可在不同机器复用 (只要 images_root 跟着移动).
    - 额外把 ``_resolve_path(images_root, img_path)`` 拼出的绝对路径写到独立字段
      ``image_path`` / ``table_image_path``, 供下游 render / VLM 等需要绝对路径
      的场景直接读取. Milvus schema 不带这个字段 -> 自动忽略.
    """
    t = item.get("type")
    cd = item.get("content") or {}
    if t == "image":
        caption = _flatten_inline(cd.get("image_caption", []))
        footnote = _flatten_inline(cd.get("image_footnote", []))
        img_path = ((cd.get("image_source") or {}).get("path") or "")
        label = _extract_caption_label(caption, "image")
        lines = [f"[Caption] {caption}" if caption else "[Image without caption]"]
        if img_path:
            lines.append(f"[Image Path] {img_path}")
        chunk: Dict[str, Any] = {
            "id": _short_id("image"),
            "type": "image",
            "section": section,
            "pages": [page_idx],
            "content": "\n".join(lines),
            "context": footnote,
            "related_assets": [],
            "_label": label,
        }
        if img_path:
            abs_path = _resolve_path(images_root, img_path)
            if abs_path:
                chunk["image_path"] = abs_path
        return _attach_bboxes(chunk, [_item_bbox(item, page_idx)])
    if t == "table":
        caption = _flatten_inline(cd.get("table_caption", []))
        footnote = _flatten_inline(cd.get("table_footnote", []))
        img_path = ((cd.get("image_source") or {}).get("path") or "")
        html = (cd.get("html") or "").strip()
        label = _extract_caption_label(caption, "table")
        lines = [f"[Caption] {caption}" if caption else "[Table without caption]"]
        if html:
            lines.append(f"[Table HTML]\n{html}")
        if img_path:
            lines.append(f"[Table Image Path] {img_path}")
        chunk = {
            "id": _short_id("table"),
            "type": "table",
            "section": section,
            "pages": [page_idx],
            "content": "\n".join(lines),
            "context": footnote,
            "related_assets": [],
            "_label": label,
        }
        if img_path:
            abs_path = _resolve_path(images_root, img_path)
            if abs_path:
                chunk["table_image_path"] = abs_path
        return _attach_bboxes(chunk, [_item_bbox(item, page_idx)])
    return None


# ---------------------------------------------------------------------------
# LLM 兜底摘要 (规则/向量/启发式都失败时调用)
# ---------------------------------------------------------------------------

def _collect_first_page_text(
    data: List[List[Dict[str, Any]]], max_chars: int = LLM_SUMMARY_MAX_INPUT_CHARS,
) -> str:
    """从第一页正文按出现顺序拼出一段文本, 喂给 LLM。"""
    if not data:
        return ""
    parts: List[str] = []
    page0 = data[0] if isinstance(data[0], list) else []
    for item in page0:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in NOISE_TYPES or t == "title":
            continue
        if t in ("paragraph", "equation_interline", "list"):
            txt = _extract_paragraph_text(item)
            if txt:
                parts.append(txt)
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def _llm_synthesize_summary(
    first_page_text: str,
    llm: "LLMClient",
    *,
    temperature: float = LLM_SUMMARY_DEFAULT_TEMPERATURE,
    max_tokens: int = LLM_SUMMARY_DEFAULT_MAX_TOKENS,
    disable_thinking: bool = LLM_SUMMARY_DEFAULT_DISABLE_THINKING,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> str:
    """让 LLM 把第一页正文压缩成一段摘要; 失败返回空串。

    调用参数可由 ``chunking.summary.llm`` 配置; 未配时使用模块默认值.
    """
    if not first_page_text or llm is None:
        return ""
    system = (system_prompt or LLM_SUMMARY_SYSTEM).strip()
    user_tpl = (user_prompt_template or LLM_SUMMARY_USER_TEMPLATE).strip()
    try:
        user = user_tpl.format(first_page_text=first_page_text)
    except KeyError as e:
        logger.warning(f"[summary-llm] user_prompt_template 占位符错误: {e}")
        return ""
    try:
        result = llm.chat(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
    except Exception as e:
        logger.warning(f"[summary-llm] 调用失败, 跳过 LLM 兜底: {e}")
        return ""
    answer = (result or {}).get("answer", "").strip()
    answer = re.sub(r"^```[a-zA-Z]*\s*", "", answer)
    answer = re.sub(r"\s*```$", "", answer).strip()
    return answer


# ---------------------------------------------------------------------------
# 主构建函数
# ---------------------------------------------------------------------------

def build_knowledge_blocks(
    data: List[List[Dict[str, Any]]],
    images_root: str = "",
    summary_title_count: int = 1,
    summary_sim_threshold: float = SUMMARY_SIM_THRESHOLD,
    embedder: Optional[EmbeddingClient] = None,
    summary_query_texts: Optional[List[str]] = None,
    split_target_chars: int = SPLIT_TARGET_CHARS,
    split_max_chars: int = SPLIT_MAX_CHARS,
    split_min_chars: int = SPLIT_MIN_CHARS,
    split_breakpoint_percentile: int = SPLIT_BREAKPOINT_PCT,
    split_length_fn: Optional[Callable[[str], int]] = None,
    split_overlap: int = 0,
    llm: Optional["LLMClient"] = None,
    doc_title: Optional[str] = None,
    # ── v6: 摘要 4 级 fallback 的 yaml-driven 参数 (None 时回退到模块默认常量) ──
    summary_title_patterns: Optional[List[re.Pattern[str]]] = None,
    summary_text_patterns: Optional[List[re.Pattern[str]]] = None,
    summary_stop_patterns: Optional[List[re.Pattern[str]]] = None,
    summary_bm25_queries: Optional[List[str]] = None,
    summary_bm25_threshold: float = SUMMARY_BM25_THRESHOLD,
    summary_bm25_enabled: bool = True,
    summary_embedding_enabled: bool = True,
    summary_max_sections: int = 2,
    summary_enabled: bool = True,
    summary_llm_enabled: bool = True,
    summary_llm_max_input_chars: int = LLM_SUMMARY_MAX_INPUT_CHARS,
    summary_llm_temperature: float = LLM_SUMMARY_DEFAULT_TEMPERATURE,
    summary_llm_max_tokens: int = LLM_SUMMARY_DEFAULT_MAX_TOKENS,
    summary_llm_disable_thinking: bool = LLM_SUMMARY_DEFAULT_DISABLE_THINKING,
    summary_llm_system_prompt: Optional[str] = None,
    summary_llm_user_template: Optional[str] = None,
    references_batch_size: int = DEFAULT_REFERENCES_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """从 content_list_v2.json 构建知识块列表 (v4: 段落级 + LLM 兜底摘要 + 显式标题)。

    摘要识别的优先级 (强信号 → 弱信号兜底已全部移除):
      1. 规则关键字: section title 命中 abstract/摘要/summary, 或前 2 页正文出现
         "Abstract:" / "摘要：" 等显式信号 + 当前 section 段首再核验.
      2. 向量相似度: section title 与三组摘要查询词的余弦相似度之和 ≥ 阈值.
      3. (无任何强信号) 由调用方传入 LLMClient → 让 LLM 把第一页正文压成摘要,
         合成一个 type=summary, section="[LLM-synthesized abstract]" 的 chunk.

    标题 (type=title) 的处理已改为: 由调用方通过 ``doc_title`` 参数显式提供
    (通常是 PDF 文件名去后缀), 注入为该文档唯一的 title chunk; 不再用启发式
    "第一页 < 50 字 text chunk" 提取.

    Args:
        data: 二维 list (按页分组)
        images_root: 图片路径的根目录
        summary_title_count: 已废弃 (仅保留参数兼容性, 不再用于弱兜底); 实际不会触发.
        summary_sim_threshold: 摘要检测阈值 (三组查询词相似度之和).
        embedder: EmbeddingClient 实例 (用于标题相似度检测和长 chunk 语义切分).
        summary_query_texts: 三组摘要查询文本, None 时使用默认 SUMMARY_QUERY_TEXTS.
        split_target_chars / split_max_chars / split_min_chars / split_breakpoint_percentile:
            长 text chunk 的语义切分参数, 详见 semantic_splitter.semantic_split.
        llm: 可选的 LLMClient; 当强信号 (规则关键字 + 向量相似度) 都未识别出摘要时,
            会用第一页正文喂给 LLM 合成准确摘要, 作为 type=summary 的 chunk 注入.
            为 None 时不做 LLM 兜底, 该文档将没有 summary chunk.
        doc_title: 可选的文献标题 (建议传 PDF 文件名去后缀); 若提供, 会在第一页内容
            最前面注入一个 type=title, content=doc_title 的 chunk; 为 None / 空串
            时不注入 title chunk.
    """
    if not isinstance(data, list):
        raise ValueError("content_list_v2 应是二维 list")

    if summary_query_texts is None:
        summary_query_texts = SUMMARY_QUERY_TEXTS
    _ = summary_title_count  # 已废弃, 显式标记为未使用

    # v6: yaml 的 summary 子表透传给 _detect_summary_sections; None 走模块默认
    if summary_enabled:
        detection = _detect_summary_sections(
            data,
            summary_sim_threshold=summary_sim_threshold,
            embedder=embedder,
            summary_query_texts=summary_query_texts,
            title_patterns=summary_title_patterns,
            text_patterns=summary_text_patterns,
            bm25_query_texts=summary_bm25_queries,
            bm25_threshold=summary_bm25_threshold,
            bm25_enabled=summary_bm25_enabled,
            embedding_enabled=summary_embedding_enabled,
            max_summary_sections=summary_max_sections,
        )
    else:
        detection = {
            "summary_sections": set(),
            "strategy": "disabled",
            "text_hit": False,
            "ordered_titles": [],
            "bm25_top": [],
            "embedding_top": [],
        }
        logger.info("[summary] chunking.summary.enabled=false, 跳过所有强信号 + LLM 检测")
    summary_sections = detection["summary_sections"]
    text_hit = bool(detection.get("text_hit"))

    chunks: List[Dict[str, Any]] = []
    figure_idx: Dict[str, str] = {}
    table_idx: Dict[str, str] = {}
    cur_section = ""
    cur_paragraphs: List[Dict[str, Any]] = []
    cur_assets: List[Dict[str, Any]] = []
    cur_pages: List[int] = []
    text_mode_triggered = False
    # MinerU reference_list 跨块/跨页累积, 统一在 section 结束或文档末尾 flush
    pending_ref_entries: List[str] = []
    pending_ref_pages: List[int] = []
    pending_ref_section = ""
    # 文档内段落计数 (1-based, 仅在 text/summary chunk 上使用)
    paragraph_counter = [0]

    def _next_para_idx() -> int:
        paragraph_counter[0] += 1
        return paragraph_counter[0]

    def _flush_pending_references() -> None:
        nonlocal pending_ref_entries, pending_ref_pages, pending_ref_section
        if not pending_ref_entries:
            return
        ref_chunks = _build_references_chunks_from_entries(
            pending_ref_entries,
            section=pending_ref_section or cur_section or "References",
            pages=pending_ref_pages,
            batch_size=references_batch_size,
        )
        chunks.extend(ref_chunks)
        pending_ref_entries = []
        pending_ref_pages = []
        pending_ref_section = ""

    def _build_section_chunks(
        groups: List[List[Dict[str, Any]]],
        section: str,
        pages: List[int],
        section_is_summary: bool,
    ) -> List[Dict[str, Any]]:
        """对一个 section 的逻辑段落组列表, 逐段构建 chunk + 长段落语义切分。"""
        # P1-5: 没有 section title 的 "preamble" 阶段 (PDF 顶部还没出现第一个 title,
        # 或者多文献一页时上一篇文章的残留 paragraph) 不消耗 paragraph_index;
        # 这样 "第 N 段" 检索路径只对真正属于本 section 的正文段计数, 不会被前置垃圾偏移.
        # 注意: 若是 abstract section, 即使 section title 为空也算正文 (走 summary 路径).
        is_preamble = (not section.strip()) and not section_is_summary
        out: List[Dict[str, Any]] = []
        for group in groups:
            # v5: equation-only group 不消耗段落号, 让正文段落编号保持连续
            # (公式被独立成 type=equation, paragraph_index=-1, 后处理建立交叉引用)
            only_equation = (
                len(group) == 1
                and group[0].get("type") == "equation_interline"
                and not section_is_summary
            )
            if only_equation or is_preamble:
                para_idx = -1
            else:
                para_idx = _next_para_idx()
            base = _build_chunk_from_group(
                group, section=section, pages=pages,
                is_summary=section_is_summary, paragraph_index=para_idx,
                summary_text_patterns=summary_text_patterns,
                summary_stop_patterns=summary_stop_patterns,
            )
            if not base:
                continue
            # preamble 阶段的非公式 chunk 标记一个 _preamble flag, 让下游 (检索/排序)
            # 可以选择性 demote; 这里**不改 type** (保持向后兼容: Milvus schema 仍认
            # text/summary/title/...), 仅在 chunk JSON 留个 "is_preamble": true 提示.
            if is_preamble and base.get("type") == "text":
                base["is_preamble"] = True
            # summary / equation 都不走语义二次切分
            if base.get("type") in ("summary", "equation"):
                out.append(base)
                continue
            split_chunks = _maybe_split_text_chunk(
                base, embedder,
                target_chars=split_target_chars,
                max_chars=split_max_chars,
                min_chars=split_min_chars,
                breakpoint_percentile=split_breakpoint_percentile,
                length_fn=split_length_fn,
                overlap_chars=split_overlap,
            )
            for sc in split_chunks:
                if "paragraph_index" not in sc:
                    sc["paragraph_index"] = para_idx
                if "bbox" not in sc and base.get("bbox"):
                    sc["bbox"] = base["bbox"]
                    sc["bboxes"] = base.get("bboxes", [])
                if is_preamble and sc.get("type") == "text":
                    sc["is_preamble"] = True
            out.extend(split_chunks)
        return out

    def _flush_section() -> None:
        nonlocal cur_paragraphs, cur_assets, cur_pages, text_mode_triggered
        nonlocal pending_ref_entries, pending_ref_pages, pending_ref_section
        # 参考文献 section (无 reference_list 时的兜底) -> 累积到 pending
        if _is_references_section(cur_section):
            entries = _extract_reference_entries(cur_paragraphs)
            if entries:
                if not pending_ref_section:
                    pending_ref_section = cur_section
                _append_pending_reference_entries(pending_ref_entries, entries)
                for p in cur_pages:
                    if p not in pending_ref_pages:
                        pending_ref_pages.append(p)
            for ac in cur_assets:
                ac.setdefault("paragraph_index", -1)
                chunks.append(ac)
            cur_paragraphs = []
            cur_assets = []
            cur_pages = []
            return

        # 强信号 1: 当前 section title 在 _detect_summary_sections 已识别为摘要
        is_summary = bool(cur_section and cur_section in summary_sections)
        # 强信号 2: 全文前两页正文里出现 "Abstract:" / "摘要：" 这种关键字时,
        #          再核验当前 section 自己的前 2 段是否也带该关键字, 双保险.
        #          这是规则关键字匹配, 不是启发式 (引言里不会这样写).
        if (
            not is_summary
            and text_hit
            and not text_mode_triggered
            and cur_pages and min(cur_pages) == 0
        ):
            head_text = "\n".join(
                _extract_paragraph_text(p) for p in cur_paragraphs[:2]
            ).strip()
            if _match_any(SUMMARY_TEXT_PATTERNS, head_text):
                is_summary = True

        # 注意: 这里**不再做任何弱启发式兜底** (前 N 个 title / 长度+句数判定),
        # 弱信号会把 introduction、related work 之类误判为 summary. 当强信号都没命中时,
        # 这个 section 一律按普通正文处理; 整个文档结束后若仍无 summary, 由 LLM 合成.

        # 1) 把当前 section 内的 items 划分为 "逻辑段落组"
        groups = _group_logical_paragraphs(cur_paragraphs)

        # 2) 摘要 section 整段合成单个 summary chunk (而不是按段落切), 这样
        #    summary 路径检索时拿到的是完整摘要, 不会被段落切碎。
        if is_summary and groups:
            merged_items: List[Dict[str, Any]] = []
            for g in groups:
                merged_items.extend(g)
            summary_chunk = _build_chunk_from_group(
                merged_items, section=cur_section, pages=cur_pages,
                is_summary=True, paragraph_index=_next_para_idx(),
                summary_text_patterns=summary_text_patterns,
                summary_stop_patterns=summary_stop_patterns,
            )
            new_chunks: List[Dict[str, Any]] = [summary_chunk] if summary_chunk else []
            text_mode_triggered = True
        else:
            new_chunks = _build_section_chunks(
                groups, section=cur_section, pages=cur_pages,
                section_is_summary=False,
            )

        asset_links: List[Dict[str, str]] = []
        for ac in cur_assets:
            asset_links.append({
                "type": ac["type"],
                "label": ac.get("_label") or ac["type"],
                "chunk_id": ac["id"],
            })

        # section 内每个 text/summary chunk 都挂上本节 asset_links (section 锚点链接):
        # 这样无论命中哪段正文, 检索侧都能把本节图/表/公式作为互补上下文带出
        # (配合检索端 assets 邻域扩展实现 "文本块 ↔ 图表块" 互补)。
        if new_chunks and asset_links:
            for nc in new_chunks:
                if nc.get("type") in ("text", "summary"):
                    nc["related_assets"] = list(asset_links)
        chunks.extend(new_chunks)

        # asset chunks 反向挂上对应 section 的第一个 text chunk 作为锚点
        anchor = new_chunks[0] if new_chunks else None
        for ac in cur_assets:
            related = list(ac.get("related_assets") or [])
            if anchor:
                related.append({
                    "type": anchor["type"],
                    "label": cur_section or "section_text",
                    "chunk_id": anchor["id"],
                })
            ac["related_assets"] = related

            section_context_parts: List[str] = []
            if cur_section:
                section_context_parts.append(f"[Section] {cur_section}")
            if ac.get("context"):
                section_context_parts.append(ac["context"])
            ac["context"] = "\n\n".join(p for p in section_context_parts if p).strip()
            # asset 不算 "段落", paragraph_index = -1
            ac.setdefault("paragraph_index", -1)
            chunks.append(ac)

        cur_paragraphs = []
        cur_assets = []
        cur_pages = []

    for page_idx, page_items in enumerate(data):
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in NOISE_TYPES:
                continue
            if t == "title":
                new_section = _extract_title_text(item)
                _flush_section()
                if pending_ref_entries and not _is_references_section(new_section):
                    _flush_pending_references()
                cur_section = new_section
                continue
            if page_idx not in cur_pages:
                cur_pages.append(page_idx)
            if t == "list" and _is_mineru_reference_list(item):
                ref_texts = _reference_list_item_texts(item)
                if _is_bibliography_reference_list(ref_texts):
                    ref_entries = _extract_entries_from_reference_list(item)
                    if ref_entries:
                        if not pending_ref_section:
                            pending_ref_section = (
                                cur_section
                                if _is_references_section(cur_section)
                                else "References"
                            )
                        _append_pending_reference_entries(
                            pending_ref_entries, ref_entries,
                        )
                        if page_idx not in pending_ref_pages:
                            pending_ref_pages.append(page_idx)
                    continue
            if t in ("paragraph", "equation_interline", "list"):
                cur_paragraphs.append(item)
            elif t in ("image", "table"):
                ac = _build_asset_chunk(item, page_idx, cur_section, images_root)
                if ac:
                    cur_assets.append(ac)
                    label = ac.get("_label")
                    if label:
                        if ac["type"] == "image":
                            figure_idx.setdefault(label, ac["id"])
                        else:
                            table_idx.setdefault(label, ac["id"])
    _flush_section()
    _flush_pending_references()

    for c in chunks:
        # references 类已经在聚合阶段决定不挂 fig/table cross-ref (引用条目里出现的
        # [Fig.X] 大概率是被引文献自身的图号, 链到本文图表反而是噪音), 直接跳过.
        if c.get("type") == "references":
            c.pop("_label", None)
            continue
        ref_text = (c.get("content") or "" if c["type"] in ("text", "summary")
                    else (c.get("content") or "") + " " + (c.get("context") or ""))
        refs = _scan_cross_refs(ref_text)
        related: List[Dict[str, str]] = []
        self_label = c.get("_label")
        for fig in refs.get("figures", []):
            if fig == self_label:
                continue
            fid = figure_idx.get(fig)
            if fid and fid != c["id"]:
                related.append({"type": "image", "label": f"Fig. {fig}", "chunk_id": fid})
        for tab in refs.get("tables", []):
            if tab == self_label:
                continue
            tid = table_idx.get(tab)
            if tid and tid != c["id"]:
                related.append({"type": "table", "label": f"Table {tab}", "chunk_id": tid})
        # 合并而非覆盖: 保留 _flush_section 建立的 section 锚点链接 (text↔asset 兜底,
        # OCR 丢失 图N/表N 编号时唯一的关联来源), 再并入显式交叉引用 (图N/表N)。
        # 按 chunk_id 去重; 显式交叉引用排在前 (更精确)。equation 双向链接随后单独追加。
        existing = c.get("related_assets") or []
        merged_assets: List[Dict[str, str]] = []
        seen_asset_ids: set = set()
        for a in list(related) + list(existing):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("chunk_id") or "").strip()
            if not aid or aid == c["id"] or aid in seen_asset_ids:
                continue
            seen_asset_ids.add(aid)
            merged_assets.append(a)
        c["related_assets"] = merged_assets
        c.pop("_label", None)

    # v5: equation chunk 与前后最近的 text/summary chunk 建立双向 related_assets,
    # 这样上下文检索时 LLM 能拿到关联公式, 公式检索时也能拿到上下文段落.
    _link_equations_to_text(chunks)

    # v6: content-driven summary promote (在 valid_summary_indexes 校验前跑)
    # 用户报的真实 case: section="" 但 content 段首 "摘要：..." 的 text chunk,
    # tier 1-3 (section title 级) 都 miss, 这里把它直接提升成 summary.
    # max_promote = summary_max_sections, 已有 summary 不再重复提升.
    if summary_enabled:
        _promote_text_to_summary(
            chunks,
            text_patterns=summary_text_patterns,
            stop_patterns=summary_stop_patterns,
            max_promote=int(summary_max_sections),
        )

    # 1. 验证 summary 必须在第一页，不在第一页的恢复为 text (兜底: 可以不取 summary)
    summary_indexes = [i for i, c in enumerate(chunks) if c.get("type") == "summary"]
    valid_summary_indexes: List[int] = []
    for idx in summary_indexes:
        c = chunks[idx]
        pages = c.get("pages") or []
        if pages and min(pages) == 0:
            valid_summary_indexes.append(idx)
        else:
            c["type"] = "text"

    # 2. 若有多个有效 summary (第一页), 仅保留最优的一个
    if len(valid_summary_indexes) > 1:
        target_len = 250

        def _summary_rank(idx: int) -> tuple:
            c = chunks[idx]
            content = (c.get("content") or "").strip()
            has_pattern = 1 if _match_any(SUMMARY_TEXT_PATTERNS, content) else 0
            len_distance = abs(len(content) - target_len)
            return (has_pattern, -len_distance, -idx)

        keep_idx = max(valid_summary_indexes, key=_summary_rank)
        for idx in valid_summary_indexes:
            if idx != keep_idx:
                chunks[idx]["type"] = "text"
        valid_summary_indexes = [keep_idx]

    # 3. (已移除启发式 title 提取; 显式 title chunk 在 LLM 兜底之后再注入,
    #    确保 [title, summary, ...content...] 的顺序)

    # 4. tier 4 — LLM 兜底摘要: 仅当 tier 1-3 都未识别出摘要 且 summary_llm_enabled=true
    #    且调用方传入了 LLMClient 时才会触发. 任何一条不满足都不调 LLM, 该文档不会有
    #    summary chunk (符合用户 "可以没有摘要, 但不能误判" 的要求).
    if valid_summary_indexes:
        logger.info(
            f"[summary] 强信号识别到 summary chunk "
            f"(tier1/1.5/2/3 之一; section_strategy={detection.get('strategy')!r}); "
            f"跳过 LLM 兜底"
        )
    elif not summary_llm_enabled:
        logger.info(
            f"[summary] tier1-3 + 段首 promote 均未命中 "
            f"(section_strategy={detection.get('strategy')!r}); "
            f"chunking.summary.llm.enabled=false, 不调 LLM 兜底, 该文档无 summary chunk"
        )
    elif llm is None:
        logger.warning(
            f"[summary] tier1-3 + 段首 promote 均未命中 "
            f"(section_strategy={detection.get('strategy')!r}), "
            f"未提供 LLMClient (检查 generation.api_key 是否配置), 该文档无 summary chunk"
        )
    else:
        first_page_text = _collect_first_page_text(data, max_chars=summary_llm_max_input_chars)
        if not first_page_text.strip():
            logger.warning(
                "[summary-llm] 第一页正文为空, 无法用 LLM 合成摘要"
            )
        else:
            logger.info(
                f"[summary-llm][tier4] tier1-3 + 段首 promote 均未命中 "
                f"(section_strategy={detection.get('strategy')!r}), "
                f"调用 LLM 合成摘要 (输入 {len(first_page_text)} 字符, "
                f"max_tokens={summary_llm_max_tokens}, "
                f"disable_thinking={summary_llm_disable_thinking})..."
            )
            synth = _llm_synthesize_summary(
                first_page_text,
                llm,
                temperature=summary_llm_temperature,
                max_tokens=summary_llm_max_tokens,
                disable_thinking=summary_llm_disable_thinking,
                system_prompt=summary_llm_system_prompt,
                user_prompt_template=summary_llm_user_template,
            )
            if not synth:
                logger.warning(
                    "[summary-llm] LLM 返回空内容或调用失败, 该文档不会有 summary chunk"
                )
            else:
                synth_chunk = {
                    "id": _short_id("summary"),
                    "type": "summary",
                    "section": "[LLM-synthesized abstract]",
                    "pages": [0],
                    "content": synth.strip(),
                    "context": "",
                    "related_assets": [],
                    "paragraph_index": 0,  # 0 表示是合成的, 不参与正文段落计数
                    "synthesized": True,
                }
                insert_at = 0
                for i, c in enumerate(chunks):
                    pages = c.get("pages") or []
                    # 把合成 summary 插到第一页其它内容前
                    if pages and min(pages) == 0:
                        insert_at = i
                        break
                else:
                    insert_at = len(chunks)
                chunks.insert(insert_at, synth_chunk)
                logger.info(
                    f"[summary-llm] LLM 合成完成 ({len(synth)} 字符), "
                    f"已注入为 summary chunk (id={synth_chunk['id']})"
                )

    # 5. 显式 title chunk: 不再走启发式提取, 改为由调用方传 doc_title
    #    (通常是 PDF 文件名去后缀), 注入为该文档唯一的 title chunk, 并固定放在第 0 位,
    #    位于 (可能存在的) summary chunk 之前.
    title_text = (doc_title or "").strip()
    if title_text:
        title_chunk = {
            "id": _short_id("title"),
            "type": "title",
            "section": "",
            "pages": [0],
            "content": title_text,
            "context": "",
            "related_assets": [],
            "paragraph_index": -1,
        }
        chunks.insert(0, title_chunk)
        logger.info(
            f"[title] 已注入 PDF 文件名作为 title chunk: {title_text!r}"
        )
    else:
        logger.info("[title] 未提供 doc_title, 该文档不会有 title chunk")

    # 确保所有 chunk 都有 paragraph_index 字段 (避免下游 KeyError)
    for c in chunks:
        c.setdefault("paragraph_index", -1)

    return chunks


# ---------------------------------------------------------------------------
# 自动发现
# ---------------------------------------------------------------------------

def autodiscover_content_list_v2(default_dir: str = "mineru_result") -> Optional[str]:
    """自动找到最新的 *_content_list_v2.json。"""
    candidates: List[str] = []
    for path in glob.glob(
        os.path.join(default_dir, "**", "*_content_list_v2.json"), recursive=True,
    ):
        candidates.append(path)
    # 也匹配无前缀的 content_list_v2.json
    for path in glob.glob(
        os.path.join(default_dir, "**", "content_list_v2.json"), recursive=True,
    ):
        if path not in candidates:
            candidates.append(path)
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return None
