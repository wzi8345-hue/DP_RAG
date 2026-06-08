"""语义切分器: 基于句子嵌入余弦距离 + 百分位阈值找自然断点。

算法参考:
- Greg Kamradt 的 5 levels of text splitting (LangChain SemanticChunker)
- chonkie SemanticChunker (向量化批处理 + 贪心合并)
- LlamaIndex SemanticSplitterNodeParser (sentence-buffer 上下文)

核心思路:
1. 多级分句 (段落 -> 句子 -> 子句), 每个最小单元尽量短小但语义完整
2. 给每个句子拼上前后 buffer 句作为 embedding 上下文 (缓解短句噪声)
3. 计算相邻句子余弦距离, 取分布的第 P 百分位作为断点阈值
4. 贪心合并: 从前往后聚合, 超过 max_chars 或遇到断点就开新块
5. 块尾再做最小长度 + 上限的二次校正

设计目标:
- 不引入额外依赖 (复用项目内的 EmbeddingClient)
- 单文档 ingest 期间一次批量 embedding 完成, 不挑剔模型
- 失败时降级到固定字符滑窗, 永不阻塞 ingest
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.embedding import EmbeddingClient

logger = logging.getLogger(__name__)

# 尺寸度量函数: 输入文本返回"大小"。默认按字符 (len); token 模式注入 estimate_tokens。
LengthFn = Callable[[str], int]


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

DEFAULT_TARGET_CHARS = 1200      # 期望块大小
DEFAULT_MAX_CHARS = 2000         # 块硬上限 (超过强制切)
DEFAULT_MIN_CHARS = 300          # 块最小阈值 (低于则与下一块合并)
DEFAULT_BREAKPOINT_PCT = 85      # 距离分布的第 N 百分位作为断点阈值
DEFAULT_SENTENCE_BUFFER = 1      # 每个句子拼前后 N 句做 embedding 上下文
DEFAULT_OVERLAP_CHARS = 0        # 相邻块的 overlap (0 = 不 overlap; >0 时按句子边界)


# ---------------------------------------------------------------------------
# token 估算 (CJK-aware 启发式, 零依赖)
# ---------------------------------------------------------------------------

# CJK 统一表意文字 + 兼容区 + 日文假名 (按 ~1 token/字符计)
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数 (面向 Qwen 类 BPE 分词的启发式, 零依赖)。

    经验近似: CJK 字符约 1 token/字符; 拉丁/数字/标点约 1 token / 4 字符。
    目的不是精确还原某个分词器, 而是给中英混排文本一个比纯字符数更一致的
    大小度量, 让 chunk 尺寸在不同语种下更均匀。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return max(1, cjk + (other + 3) // 4)


def _resolve_len(length_fn: Optional[LengthFn]) -> LengthFn:
    """返回尺寸度量函数; None -> 字符数 len。"""
    return length_fn if length_fn is not None else len


# ---------------------------------------------------------------------------
# 多级分句: 段落 -> 句子 -> 子句, 直到每段 <= max_unit_chars
# ---------------------------------------------------------------------------

# 顺序: 强 -> 弱; 越前面越优先在该粒度切
_SPLIT_LEVELS = [
    re.compile(r"\n{2,}"),                # 段落
    re.compile(r"(?<=[。！？!?])\s*"),    # 句末强标点 (保留标点)
    re.compile(r"(?<=[；;])\s*"),         # 分号
    re.compile(r"(?<=[，,])\s*"),         # 逗号 (最后兜底)
]


def _split_by_levels(text: str, max_unit_chars: int) -> List[str]:
    """递归多级切分, 直到每段 <= max_unit_chars。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_unit_chars:
        return [text]
    for pattern in _SPLIT_LEVELS:
        parts = [p.strip() for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            out: List[str] = []
            for p in parts:
                if len(p) <= max_unit_chars:
                    out.append(p)
                else:
                    out.extend(_split_by_levels(p, max_unit_chars))
            return out
    # 无标点可切, 硬切字符
    return [text[i:i + max_unit_chars] for i in range(0, len(text), max_unit_chars)]


# ---------------------------------------------------------------------------
# 余弦距离 + 百分位
# ---------------------------------------------------------------------------

def _cosine_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    sim = dot / (na * nb)
    # 数值噪声裁剪
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# ---------------------------------------------------------------------------
# 主切分函数
# ---------------------------------------------------------------------------

@dataclass
class SemanticChunk:
    """语义切分输出。"""
    text: str
    sentence_indices: List[int]  # 原始句子下标范围 (用于调试)


def semantic_split(
    text: str,
    embedder: Optional["EmbeddingClient"],
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    breakpoint_percentile: int = DEFAULT_BREAKPOINT_PCT,
    sentence_buffer: int = DEFAULT_SENTENCE_BUFFER,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    length_fn: Optional[LengthFn] = None,
) -> List[SemanticChunk]:
    """对单段长文本做语义切分, 返回多个 chunk。

    Args:
        text: 输入长文本
        embedder: EmbeddingClient 实例; 为 None 时降级到等长滑窗
        target_chars: 期望每个 chunk 的大小 (单位由 length_fn 决定, 默认字符)
        max_chars: 硬上限 (同上单位)
        min_chars: 最小长度 (低于则与下一块合并; 同上单位)
        breakpoint_percentile: 取相邻句子距离分布的第 N 百分位作为切分阈值
        sentence_buffer: embedding 时给每个句子拼前后 N 句做上下文
        overlap_chars: 相邻 chunk 的 overlap 预算 (0 = 无; >0 时按句子边界回填)
        length_fn: 尺寸度量函数; None=按字符 len, 传 estimate_tokens 则按 token 计

    Returns:
        SemanticChunk 列表; 输入足够短时返回单元素列表
    """
    _len = _resolve_len(length_fn)
    text = (text or "").strip()
    if not text:
        return []
    if _len(text) <= max_chars:
        return [SemanticChunk(text=text, sentence_indices=[0])]

    # 1) 多级分句, 单元上限取 target_chars 的一半 (避免单句撑爆 chunk)
    sentences = _split_by_levels(text, max_unit_chars=max(200, target_chars // 2))
    n = len(sentences)
    if n <= 1:
        # 无法再细分: 只能滑窗
        return _fallback_window_split(text, target_chars, max_chars, overlap_chars)

    # 2) embedding 失败 / 未提供 -> 滑窗兜底
    if embedder is None:
        return _greedy_merge_no_embedding(
            sentences, target_chars, max_chars, min_chars, length_fn=length_fn,
        )

    # 3) 给每个句子拼 buffer 上下文做 embedding
    buffered_texts: List[str] = []
    for i in range(n):
        lo = max(0, i - sentence_buffer)
        hi = min(n, i + sentence_buffer + 1)
        buffered_texts.append(" ".join(sentences[lo:hi]))

    try:
        vectors = embedder.embed_all(buffered_texts)
    except Exception as e:
        logger.warning(f"[semantic_splitter] embedding 失败, 降级到等长滑窗: {e}")
        return _greedy_merge_no_embedding(
            sentences, target_chars, max_chars, min_chars, length_fn=length_fn,
        )

    if len(vectors) != n:
        logger.warning(
            f"[semantic_splitter] 返回向量数 {len(vectors)} != 句数 {n}, 降级到等长滑窗"
        )
        return _greedy_merge_no_embedding(
            sentences, target_chars, max_chars, min_chars, length_fn=length_fn,
        )

    # 4) 相邻余弦距离 + 百分位阈值
    distances = [
        _cosine_distance(vectors[i], vectors[i + 1]) for i in range(n - 1)
    ]
    if not distances:
        return [SemanticChunk(text=text, sentence_indices=list(range(n)))]
    threshold = _percentile(distances, breakpoint_percentile)

    # 5) 贪心合并: 累计到 target_chars 或遇断点就关闭当前 chunk
    chunks: List[SemanticChunk] = []
    cur: List[str] = []
    cur_indices: List[int] = []
    cur_len = 0
    for i, sent in enumerate(sentences):
        cur.append(sent)
        cur_indices.append(i)
        cur_len += _len(sent)
        is_breakpoint = (i < n - 1) and (distances[i] >= threshold)
        if cur_len >= max_chars or (cur_len >= target_chars and is_breakpoint):
            chunks.append(SemanticChunk(
                text=_join_sentences(cur),
                sentence_indices=list(cur_indices),
            ))
            cur, cur_indices, cur_len = [], [], 0
    if cur:
        chunks.append(SemanticChunk(
            text=_join_sentences(cur),
            sentence_indices=list(cur_indices),
        ))

    # 6) 二次校正: 太短的 chunk 与下一块合并 (除非已是最后一块)
    chunks = _merge_short_chunks(chunks, min_chars, max_chars, length_fn=length_fn)

    # 7) 添加 overlap (可选, 按句子边界回填, 不在句中截断)
    if overlap_chars > 0 and len(chunks) > 1:
        chunks = _apply_overlap(
            chunks, sentences, overlap_chars, max_chars, length_fn=length_fn,
        )

    return chunks


# ---------------------------------------------------------------------------
# 辅助 / 兜底逻辑
# ---------------------------------------------------------------------------

def _join_sentences(sents: List[str]) -> str:
    """拼接句子: 中文不加空格, 已有标点直接连接。"""
    return "".join(s if i == 0 else (s if _starts_with_zh(s) else " " + s)
                   for i, s in enumerate(sents)).strip()


def _starts_with_zh(s: str) -> bool:
    if not s:
        return False
    c = s[0]
    return "\u4e00" <= c <= "\u9fff"


def _merge_short_chunks(
    chunks: List[SemanticChunk], min_chars: int, max_chars: int,
    length_fn: Optional[LengthFn] = None,
) -> List[SemanticChunk]:
    _len = _resolve_len(length_fn)
    if not chunks:
        return chunks
    out: List[SemanticChunk] = []
    for c in chunks:
        if out and _len(out[-1].text) < min_chars and \
                _len(out[-1].text) + _len(c.text) <= max_chars:
            prev = out[-1]
            merged_text = prev.text + (
                "" if _starts_with_zh(c.text) else " "
            ) + c.text
            out[-1] = SemanticChunk(
                text=merged_text.strip(),
                sentence_indices=prev.sentence_indices + c.sentence_indices,
            )
        else:
            out.append(c)
    # 尾部太短 -> 与前一块合并
    if len(out) >= 2 and _len(out[-1].text) < min_chars and \
            _len(out[-2].text) + _len(out[-1].text) <= max_chars:
        last = out.pop()
        prev = out[-1]
        merged_text = prev.text + (
            "" if _starts_with_zh(last.text) else " "
        ) + last.text
        out[-1] = SemanticChunk(
            text=merged_text.strip(),
            sentence_indices=prev.sentence_indices + last.sentence_indices,
        )
    return out


def _apply_overlap(
    chunks: List[SemanticChunk], sentences: List[str],
    overlap_chars: int, max_chars: int,
    length_fn: Optional[LengthFn] = None,
) -> List[SemanticChunk]:
    """按句子边界回填 overlap: 把前一块尾部的若干完整句子前置到当前块。

    不在句子中间截断; 若加上 overlap 会超过 max_chars 则跳过该块的 overlap。
    sentence_indices 沿用当前块自身的下标 (overlap 句归属上一块, 仅用于上下文)。
    """
    _len = _resolve_len(length_fn)
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        cur = chunks[i]
        prev = chunks[i - 1]
        tail: List[str] = []
        budget = 0
        for sidx in reversed(prev.sentence_indices):
            if sidx < 0 or sidx >= len(sentences):
                continue
            s = sentences[sidx]
            if tail and budget + _len(s) > overlap_chars:
                break
            tail.insert(0, s)
            budget += _len(s)
            if budget >= overlap_chars:
                break
        overlap_text = _join_sentences(tail) if tail else ""
        if overlap_text:
            new_text = _join_sentences([overlap_text, cur.text])
        else:
            new_text = cur.text
        if _len(new_text) > max_chars:
            new_text = cur.text  # overlap 会超上限 -> 放弃 overlap, 保留原块
        out.append(SemanticChunk(
            text=new_text, sentence_indices=cur.sentence_indices,
        ))
    return out


def _greedy_merge_no_embedding(
    sentences: List[str], target_chars: int, max_chars: int, min_chars: int,
    length_fn: Optional[LengthFn] = None,
) -> List[SemanticChunk]:
    """无 embedding 兜底: 按 target_chars 贪心合并句子。"""
    _len = _resolve_len(length_fn)
    chunks: List[SemanticChunk] = []
    cur: List[str] = []
    cur_indices: List[int] = []
    cur_len = 0
    for i, s in enumerate(sentences):
        cur.append(s)
        cur_indices.append(i)
        cur_len += _len(s)
        if cur_len >= target_chars:
            chunks.append(SemanticChunk(
                text=_join_sentences(cur), sentence_indices=list(cur_indices),
            ))
            cur, cur_indices, cur_len = [], [], 0
    if cur:
        chunks.append(SemanticChunk(
            text=_join_sentences(cur), sentence_indices=list(cur_indices),
        ))
    return _merge_short_chunks(chunks, min_chars, max_chars, length_fn=length_fn)


def _fallback_window_split(
    text: str, target_chars: int, max_chars: int, overlap_chars: int,
) -> List[SemanticChunk]:
    """最末兜底: 等长字符滑窗。"""
    step = max(1, target_chars - overlap_chars)
    out: List[SemanticChunk] = []
    i = 0
    idx = 0
    while i < len(text):
        seg = text[i:i + max_chars]
        out.append(SemanticChunk(text=seg, sentence_indices=[idx]))
        i += step
        idx += 1
    return out
