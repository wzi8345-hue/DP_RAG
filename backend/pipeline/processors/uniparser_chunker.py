"""UniParser 解析结果 -> knowledge blocks 转换 (v5 schema)。

输入: ``uniparser_result.json`` (UniParser ``/get-formatted`` 的原始响应), 用其中的
``pages_dict`` (按页扁平 layout) 作为主输入. 不用 ``content`` (含 base64 图,
体积巨大), 不用 ``pages_tree`` (group/figuregroup 的层级对 chunker 没增值).

输出 chunk schema 与 MinerU chunker 完全一致 (兼容现有 Milvus / vectorizer /
retriever): ``id`` / ``type`` / ``section`` / ``pages`` / ``content`` / ``context`` /
``related_assets`` / ``paragraph_index`` (+ 可选 ``parent_chunk_id`` /
``chunk_index`` / ``chunk_total``).

主线策略对齐 MinerU chunker, 但针对 UniParser 实际产物做了以下调整:

1. **块预处理**: 按 ``(page, order)`` 全局排序 ``pages_dict`` 内所有 block, 形成
   线性阅读序; ``hidden=true`` / ``conf < min_conf`` / NOISE_TYPES 直接丢弃.

2. **section 推断**: ``documenttitle`` 用于 doc title; ``title`` 块的文本作为
   当前 section 名 (UniParser 没给标题层级, 自动按 "3.1." 这种数字前缀推断 level).

3. **block -> chunk 类型映射**:
   - ``paragraph``                 -> ``text`` (段落级, 长段走 semantic_split)
   - ``equation``                  -> ``equation`` (content = ``$$ {latex_repr} $$``)
   - ``table``                     -> ``table`` (HTML 取自 ``structure`` 字段;
     content 用 ``[Caption] xxx\n[Table HTML]\n<html>`` 与 MinerU 对齐, 让
     vectorizer 的 _TABLE_HTML_BLOCK_RE 直接吃)
   - ``figure``/``image``/``chart``-> ``image`` (caption 来自附近的
     ``imagecaption``/``legend`` 块; 没有图片落盘, 只有 [Caption])
   - ``reference``                 -> 聚合 batch -> ``references``
   - ``imagecaption``/``tablecaption``/``equationid``/``legend``
                                   -> 不单独成 chunk, 作为附近 figure/table/equation 的
                                      caption 来源
   - ``documenttitle``             -> 注入唯一 ``title`` chunk
   - ``title``                     -> 只更新 cur_section, 不出 chunk (与 MinerU 一致)
   - NOISE: ``pageheader`` / ``pagefooter`` / ``pagenumber`` / ``pagenote`` /
     ``watermark`` / ``hline``    -> 直接丢

4. **摘要 (summary) 检测**: 与 MinerU 同一套 (规则关键字 → 向量相似度 → LLM 兜底);
   把 paragraph 转成与 MinerU content_list_v2 兼容的 mock dict 复用现成函数,
   避免重写一份强信号检测.

5. **equation 双向 link**: 直接复用 MinerU 的 ``_link_equations_to_text``.

6. **figure/table 交叉引用**: 复用 ``_scan_cross_refs`` (扫 ``Fig. N`` / ``Table N``
   字面量), 给 text/summary chunk 自动挂 related_assets.

7. **元数据 sidecar**: chunker 输出时一并写 ``knowledge_blocks_meta.json``,
   让现有 ``MilvusIngester._load_meta_sidecar`` 直接复用 (doc_id / doc_name /
   publication_year), 不必额外改 ingest 逻辑.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..clients.embedding import EmbeddingClient
from .chunker import (
    _detect_summary_sections,
    _promote_text_to_summary,
    _link_equations_to_text,
    _llm_synthesize_summary,
    _maybe_split_text_chunk,
    _scan_cross_refs,
    _short_id,
    _extract_caption_label,
    _strip_summary_prefix,
    _clip_summary_text,
    is_garbled_text,
    _normalize_text,
    sanitize_section,
    # 对齐 MinerU 正文质量: 逻辑段落合并 (含误切合并/短段合并) + 期刊元数据行过滤
    _is_metadata_line,
    _group_logical_paragraphs,
    _group_combined_text,
    SUMMARY_QUERY_TEXTS,
    SUMMARY_SIM_THRESHOLD,
    REFERENCES_SECTION_PATTERNS,
    DEFAULT_REFERENCES_BATCH_SIZE,
    LLM_SUMMARY_MAX_INPUT_CHARS,
    LLM_SUMMARY_DEFAULT_TEMPERATURE,
    LLM_SUMMARY_DEFAULT_MAX_TOKENS,
    LLM_SUMMARY_DEFAULT_DISABLE_THINKING,
)
from .semantic_splitter import (
    DEFAULT_TARGET_CHARS,
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_BREAKPOINT_PCT,
)

if TYPE_CHECKING:
    from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)

# ── 类型集合 ─────────────────────────────────────────────────────────────

# UniParser block 类型 -> 处理策略
NOISE_TYPES = {
    "pageheader", "pagefooter", "pagenumber",
    "pagenote", "watermark", "hline",
}
# 这些 type 出 chunk
TEXT_LIKE_TYPES = {"paragraph"}                  # 普通正文段
REFERENCE_TYPE = "reference"                     # 单条参考文献
TITLE_TYPE = "title"                             # 章节标题 (只更新 cur_section)
DOC_TITLE_TYPE = "documenttitle"                 # 文献主标题 -> title chunk
EQUATION_TYPE = "equation"                       # 独立公式
TABLE_TYPE = "table"                             # 表
IMAGECAPTION_TYPE = "imagecaption"               # 图标题 -> 锚定 1 个 image chunk
# 图像视觉块: 不单独成 chunk (会让 1 张多面板图被拆成多个无意义 chunk),
# 它们的 "语义" 由邻近的 imagecaption 承载. 没 imagecaption 的孤儿图直接丢.
FIGURE_VISUAL_TYPES = {"figure", "image", "chart"}
# Caption/legend 类: 不单独成 chunk
# - imagecaption 在专门的 image 锚点循环里消耗
# - tablecaption / equationid 在 table / equation 锚点里就近查找
# - legend 在 UniParser 里基本是 "(a)/(b)" 这种子面板标签, 视为噪音直接丢
CAPTION_TYPES = {"imagecaption", "tablecaption", "equationid"}
LEGEND_TYPE = "legend"
# 容器: 在 pages_dict 里也会出现 (figuregroup 重复了很多次), 直接跳, 内容由
# 子块负担 (我们用 pages_dict 而不是 pages_tree, 容器纯粹是冗余信号).
CONTAINER_TYPES = {"group", "figuregroup"}

# section title 文本里的数字前缀, 用来推断 heading level (1. / 1.1 / 1.1.1)
_SECTION_NUM_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)[\s.]")


def _block_text(b: Dict[str, Any]) -> str:
    """从 UniParser block 取纯文本; 不存在或空返回 ''。"""
    t = b.get("text")
    if isinstance(t, str):
        return _normalize_text(t.strip())
    return ""


def _is_drop_block(b: Dict[str, Any], min_conf: float) -> bool:
    """判断是否丢弃这个 block (NOISE / hidden / 低置信)。"""
    if not isinstance(b, dict):
        return True
    if b.get("type") in NOISE_TYPES:
        return True
    if b.get("hidden") is True:
        return True
    conf = b.get("conf")
    if isinstance(conf, (int, float)) and conf < min_conf:
        return True
    return False


def _collect_all_blocks(
    pages_dict: List[List[Dict[str, Any]]], min_conf: float,
) -> List[Dict[str, Any]]:
    """把 pages_dict 摊平 + 按 (page, order) 排序 + 丢噪声 + 去 container 重复。

    container (group / figuregroup) 在 pages_dict 里会有多份冗余, 直接全部跳过 —
    它们的子块 (figure/image/chart/equation/table) 在 pages_dict 里也是独立条目,
    不会丢内容.
    """
    out: List[Dict[str, Any]] = []
    seen_blocks: set = set()  # 防 (page, block_id) 重复 (UniParser 偶有重复条目)
    for page in pages_dict or []:
        if not isinstance(page, list):
            continue
        for b in page:
            if _is_drop_block(b, min_conf):
                continue
            t = b.get("type")
            if t in CONTAINER_TYPES:
                continue
            key = (b.get("page"), b.get("block"))
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            out.append(b)
    # 按 (page, order) 全局排序; page 是 0-based, order 是页内阅读序
    out.sort(key=lambda b: (int(b.get("page", 0)), int(b.get("order", 0))))
    return out


def _collect_all_blocks_from_tree(
    pages_tree: List[Any], min_conf: float,
) -> List[Dict[str, Any]]:
    """从 pages_tree 中 DFS 收集叶子 block, **保留 group 归属信息**。

    pages_tree 与 pages_dict 的关键差异: 顶层 ``group`` / ``figuregroup`` /
    ``tablegroup`` 节点用 ``items`` 包住一组逻辑相关的子块. 例如一张多面板
    复合图 (Fig. 1 含 (a)(b)(c)(d)) 在 tree 里是 1 个 figuregroup, items 里
    放 4 个 figure + 1 个 imagecaption; 而在 pages_dict 里它们是 4+1 个独立
    blocks, 没有归属关系.

    本函数在 DFS 时:
    - 跳过 NOISE / hidden / low-conf / 容器自身 (与 pages_dict 一致)
    - 遇到叶子 block 时, **在它身上挂一个 ``_group_id`` 字段**, 值为最近的
      祖先容器 (group / figuregroup / tablegroup) 的 ``(page, block)`` 元组;
      没有祖先容器时为 None.
    - 输出按 (page, order) 全局排序

    chunker 后续看到 ``_group_id`` 可以做更精确的去重 + caption 锚定 (例如
    同 group 的多个 figure 子块只出 1 个 image chunk).
    """
    out: List[Dict[str, Any]] = []
    seen_blocks: set = set()

    def _walk(node: Any, group_id: Optional[tuple]) -> None:
        if not isinstance(node, dict):
            return
        t = node.get("type")
        # 是容器: 不出叶子, 继续向下递归, group_id 更新为自己
        if t in CONTAINER_TYPES:
            new_gid = (node.get("page"), node.get("block"))
            for child in (node.get("items") or []):
                _walk(child, new_gid)
            return
        # 叶子: 应用过滤
        if _is_drop_block(node, min_conf):
            return
        key = (node.get("page"), node.get("block"))
        if key in seen_blocks:
            return
        seen_blocks.add(key)
        # 浅拷贝 + 挂 _group_id (不污染原 dict)
        leaf = dict(node)
        leaf["_group_id"] = group_id
        out.append(leaf)
        # 防御: 叶子也可能有 items (不常见, 但 figure 在某些版本会带子 image)
        for child in (node.get("items") or []):
            _walk(child, group_id)

    for page in pages_tree or []:
        if not isinstance(page, list):
            continue
        for top in page:
            _walk(top, None)

    out.sort(key=lambda b: (int(b.get("page", 0)), int(b.get("order", 0))))
    return out


def _section_level(title: str) -> int:
    """根据数字前缀推断 heading level. 没数字前缀返回 1。"""
    m = _SECTION_NUM_PREFIX_RE.match(title or "")
    if not m:
        return 1
    return len(m.group(1).split("."))


def _table_caption_for(
    block: Dict[str, Any], block_idx: int, all_blocks: List[Dict[str, Any]],
    radius: int = 3,
) -> Tuple[str, str]:
    """找 table block 附近的 caption + label.

    Returns:
        (caption, label)

    备注: UniParser 实际产物里 legend 几乎都是 "(a)/(b)" 子标签 (不是表注),
    所以这里不再尝试抓 footnote. 若以后样本里 legend 真出现表注内容, 可以
    在 build_image_chunk 同样的 radius scan 里加回来.
    """
    page = block.get("page")
    for d in range(1, radius + 1):
        for sign in (-1, 1):
            j = block_idx + sign * d
            if 0 <= j < len(all_blocks):
                cand = all_blocks[j]
                if cand.get("page") == page and cand.get("type") == "tablecaption":
                    txt = _block_text(cand)
                    if txt:
                        label = _extract_caption_label(txt, "table") or ""
                        return txt, label
    return "", ""


def _equation_caption_for(
    block: Dict[str, Any], block_idx: int, all_blocks: List[Dict[str, Any]],
    radius: int = 2,
) -> str:
    """找 equation 后面紧跟的 equationid (公式编号), 拼到 latex 前面方便检索."""
    page = block.get("page")
    for d in range(1, radius + 1):
        j = block_idx + d
        if 0 <= j < len(all_blocks):
            cand = all_blocks[j]
            if cand.get("page") == page and cand.get("type") == "equationid":
                txt = _block_text(cand)
                if txt:
                    return txt
    return ""


# ── chunk 构建 ─────────────────────────────────────────────────────────


def _build_paragraph_chunk(
    text: str, section: str, page: int, paragraph_index: int,
) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    return {
        "id": _short_id("text"),
        "type": "text",
        "section": section,
        "pages": [int(page)],
        "content": text.strip(),
        "context": "",
        "related_assets": [],
        "paragraph_index": paragraph_index,
    }


def _build_equation_chunk(
    latex: str, section: str, page: int, eq_label: str = "",
) -> Optional[Dict[str, Any]]:
    """latex 已剥过 $$ 时自动包一层, 已带 $$ 不重复。"""
    if not latex or not latex.strip():
        return None
    body = latex.strip()
    if not body.startswith("$$"):
        body = f"$$\n{body}\n$$"
    content = body if not eq_label else f"{eq_label}\n{body}"
    return {
        "id": _short_id("equation"),
        "type": "equation",
        "section": section,
        "pages": [int(page)],
        "content": content,
        "context": "",
        "related_assets": [],
        "paragraph_index": -1,
    }


def _build_table_chunk(
    block: Dict[str, Any], section: str, caption: str, label: str,
) -> Optional[Dict[str, Any]]:
    """与 MinerU 的 table chunk 完全同形式: content = [Caption] ...\n[Table HTML]\n<html>"""
    structure = (block.get("structure") or "").strip()
    if not structure and not caption:
        return None
    lines: List[str] = []
    lines.append(f"[Caption] {caption}" if caption else "[Table without caption]")
    if structure:
        lines.append(f"[Table HTML]\n{structure}")
    return {
        "id": _short_id("table"),
        "type": "table",
        "section": section,
        "pages": [int(block.get("page", 0))],
        "content": "\n".join(lines),
        "context": "",
        "related_assets": [],
        "paragraph_index": -1,
        "_label": label,
    }


def _build_image_chunk_from_caption(
    caption_block: Dict[str, Any], section: str,
) -> Optional[Dict[str, Any]]:
    """以 imagecaption 为锚点构 image chunk。

    UniParser 不自动切图, 第一期 image chunk 只承载 caption 文本 (不带 [Image Path]).
    label 用 _extract_caption_label 从 "Fig. N ..." 抽出来, 让现有 cross-ref 复用.

    Returns None when caption is empty/garbage — 我们不为没 caption 的视觉块兜底,
    那种孤儿图在 RAG 里没有语义价值.
    """
    caption = _block_text(caption_block)
    if not caption or len(caption) < 4:
        return None
    label = _extract_caption_label(caption, "image") or ""
    return {
        "id": _short_id("image"),
        "type": "image",
        "section": section,
        "pages": [int(caption_block.get("page", 0))],
        "content": f"[Caption] {caption}",
        "context": "",
        "related_assets": [],
        "paragraph_index": -1,
        "_label": label,
    }


def _build_doc_title_chunk(
    text: str, page: int = 0,
) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    return {
        "id": _short_id("title"),
        "type": "title",
        "section": "",
        "pages": [int(page)],
        "content": text.strip(),
        "context": "",
        "related_assets": [],
        "paragraph_index": -1,
    }


def _build_references_chunks_from_blocks(
    ref_blocks: List[Dict[str, Any]],
    section: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """把 UniParser 的 reference 类 block 按 batch 聚合为 references chunk。"""
    entries: List[str] = []
    pages_seen: set = set()
    for b in ref_blocks:
        txt = _block_text(b)
        if not txt:
            continue
        entries.append(txt)
        pages_seen.add(int(b.get("page", -1)))
    if not entries:
        return []
    sorted_pages = sorted(p for p in pages_seen if p >= 0)
    out: List[Dict[str, Any]] = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        content = "\n\n".join(batch).strip()
        if not content:
            continue
        out.append({
            "id": _short_id("references"),
            "type": "references",
            "section": section or "References",
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


# ── 摘要检测: 复用 MinerU 的 _detect_summary_sections (需要 mock content_list_v2 输入) ──


def _to_mineru_like(
    all_blocks: List[Dict[str, Any]], page_count: int,
) -> List[List[Dict[str, Any]]]:
    """把 UniParser 扁平 blocks 转成 MinerU content_list_v2 风格的二维 list,
    只填 _detect_summary_sections / _collect_first_page_text 用到的字段。

    映射:
      title       -> {type: 'title', content: {title_content: [{type:'text', content:text}]}}
      paragraph   -> {type: 'paragraph', content: {paragraph_content: [{type:'text', content:text}]}}
      其它        -> 忽略 (摘要检测只看 title + paragraph)
    """
    pages: List[List[Dict[str, Any]]] = [[] for _ in range(max(page_count, 1))]
    for b in all_blocks:
        t = b.get("type")
        page = int(b.get("page", 0))
        if page < 0 or page >= len(pages):
            continue
        text = _block_text(b)
        if not text:
            continue
        if t == TITLE_TYPE:
            pages[page].append({
                "type": "title",
                "content": {"title_content": [{"type": "text", "content": text}]},
            })
        elif t in TEXT_LIKE_TYPES:
            pages[page].append({
                "type": "paragraph",
                "content": {
                    "paragraph_content": [{"type": "text", "content": text}],
                },
            })
    return pages


def _collect_first_page_text_uniparser(
    all_blocks: List[Dict[str, Any]], max_chars: int = LLM_SUMMARY_MAX_INPUT_CHARS,
) -> str:
    """从首页正文按出现顺序拼出一段, 喂给 LLM 兜底摘要."""
    parts: List[str] = []
    for b in all_blocks:
        if int(b.get("page", -1)) != 0:
            continue
        if b.get("type") not in TEXT_LIKE_TYPES:
            continue
        txt = _block_text(b)
        if txt:
            parts.append(txt)
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


# ── 主入口 ────────────────────────────────────────────────────────────


def build_knowledge_blocks_uniparser(
    result_json: Dict[str, Any],
    doc_title: Optional[str] = None,
    summary_sim_threshold: float = SUMMARY_SIM_THRESHOLD,
    embedder: Optional[EmbeddingClient] = None,
    summary_query_texts: Optional[List[str]] = None,
    split_target_chars: int = DEFAULT_TARGET_CHARS,
    split_max_chars: int = DEFAULT_MAX_CHARS,
    split_min_chars: int = DEFAULT_MIN_CHARS,
    split_breakpoint_percentile: int = DEFAULT_BREAKPOINT_PCT,
    split_length_fn: Optional[Callable[[str], int]] = None,
    split_overlap: int = 0,
    llm: Optional["LLMClient"] = None,
    references_batch_size: int = DEFAULT_REFERENCES_BATCH_SIZE,
    min_conf: float = 0.5,
    source: str = "pages_dict",
    # ── v6: 摘要 4 级 fallback 的 yaml-driven 参数 (与 MinerU chunker 一致) ──
    summary_title_patterns: Optional[List["re.Pattern[str]"]] = None,
    summary_text_patterns: Optional[List["re.Pattern[str]"]] = None,
    summary_stop_patterns: Optional[List["re.Pattern[str]"]] = None,
    summary_bm25_queries: Optional[List[str]] = None,
    summary_bm25_threshold: Optional[float] = None,
    summary_bm25_enabled: bool = True,
    summary_embedding_enabled: bool = True,
    summary_max_sections: int = 2,
    summary_enabled: bool = True,
    summary_llm_enabled: bool = True,
    summary_llm_max_input_chars: int = 6000,
    summary_llm_temperature: float = LLM_SUMMARY_DEFAULT_TEMPERATURE,
    summary_llm_max_tokens: int = LLM_SUMMARY_DEFAULT_MAX_TOKENS,
    summary_llm_disable_thinking: bool = LLM_SUMMARY_DEFAULT_DISABLE_THINKING,
    summary_llm_system_prompt: Optional[str] = None,
    summary_llm_user_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """把 ``uniparser_result.json`` 转为 v5 knowledge blocks.

    Args:
        source: chunker 的数据源, 决定从 result_json 的哪个字段取 block.
            - ``"pages_dict"`` (默认): 按页扁平 layout. 简单稳, 但 figuregroup
              里多个子面板各自是独立 block, 我们用 imagecaption 做锚点来去重.
            - ``"pages_tree"``: 树形 layout. DFS 摊平时**保留 group_id**,
              chunker 据此做更精确的 group 级去重 (1 个 figuregroup 永远只
              出 1 个 image chunk, 即便它有 2 个 imagecaption 或 0 个 caption).
              想要的话, 在 yaml 设 ``chunking.uniparser.source: pages_tree``,
              并确保 UniParser 调用时开了 ``output_flags.pages_tree: true``.

    输出与 ``build_knowledge_blocks`` (MinerU) 同形, 可直接进现有 vectorizer + Milvus.
    """
    if summary_query_texts is None:
        summary_query_texts = SUMMARY_QUERY_TEXTS

    source = (source or "pages_dict").strip().lower()
    if source not in ("pages_dict", "pages_tree"):
        raise ValueError(
            f"chunking.uniparser.source 必须是 pages_dict / pages_tree, 当前: {source!r}"
        )

    if source == "pages_tree":
        pages_tree = result_json.get("pages_tree") or []
        if not isinstance(pages_tree, list) or not pages_tree:
            # 兜底: yaml 设的是 pages_tree 但 API 没返回 (用户忘开 output_flags)
            logger.warning(
                "[uniparser-chunker] source=pages_tree 但 result_json 里没有 pages_tree, "
                "回退到 pages_dict; 请检查 yaml 是否设了 uniparser.output_flags.pages_tree: true"
            )
            source = "pages_dict"
        else:
            all_blocks = _collect_all_blocks_from_tree(pages_tree, min_conf=min_conf)
            logger.info(
                f"[uniparser-chunker] source=pages_tree, DFS 摊平后 "
                f"{len(all_blocks)} blocks (含 _group_id)"
            )

    if source == "pages_dict":
        pages_dict = result_json.get("pages_dict") or []
        if not isinstance(pages_dict, list):
            raise ValueError("uniparser_result.json 缺少 pages_dict 或格式不对")
        all_blocks = _collect_all_blocks(pages_dict, min_conf=min_conf)
        logger.info(
            f"[uniparser-chunker] source=pages_dict, {len(all_blocks)} blocks"
        )

    if not all_blocks:
        logger.warning(f"[uniparser-chunker] {source} 没有可用 block")
        return []

    page_count = max(int(b.get("page", 0)) for b in all_blocks) + 1

    # 用 MinerU 的 4 级 fallback 强信号检测 (rule → bm25 → embedding → none, LLM 兜底在后面)
    if summary_enabled:
        mineru_like = _to_mineru_like(all_blocks, page_count)
        det_kwargs: Dict[str, Any] = {
            "summary_sim_threshold": summary_sim_threshold,
            "embedder": embedder,
            "summary_query_texts": summary_query_texts,
            "max_summary_sections": summary_max_sections,
        }
        if summary_title_patterns is not None:
            det_kwargs["title_patterns"] = summary_title_patterns
        if summary_text_patterns is not None:
            det_kwargs["text_patterns"] = summary_text_patterns
        if summary_bm25_queries is not None:
            det_kwargs["bm25_query_texts"] = summary_bm25_queries
        if summary_bm25_threshold is not None:
            det_kwargs["bm25_threshold"] = summary_bm25_threshold
        det_kwargs["bm25_enabled"] = summary_bm25_enabled
        det_kwargs["embedding_enabled"] = summary_embedding_enabled
        detection = _detect_summary_sections(mineru_like, **det_kwargs)
        summary_sections: set = set(detection["summary_sections"] or [])
        text_hit = bool(detection.get("text_hit"))
        logger.info(
            f"[uniparser-chunker] summary detection: strategy={detection.get('strategy')!r} "
            f"sections={summary_sections} text_hit={text_hit}"
        )
    else:
        detection = {"strategy": "disabled"}
        summary_sections = set()
        text_hit = False
        logger.info(
            "[uniparser-chunker] chunking.summary.enabled=false, 跳过所有强信号 + LLM 检测"
        )

    chunks: List[Dict[str, Any]] = []
    figure_idx: Dict[str, str] = {}
    table_idx: Dict[str, str] = {}
    # source=pages_tree 时: 同 group_id 只出 1 个 image / table chunk,
    # 即使 group 内有多个 imagecaption (少见但发生过) 也只取第一个.
    seen_image_groups: set = set()
    seen_table_groups: set = set()

    # 注入文献 title chunk: 优先用 UniParser 解析出的真实标题 (documenttitle),
    # 仅当其缺失/乱码时才回退到 doc_title 参数 (通常为 PDF 文件名去后缀)。
    documenttitle_text = ""
    for b in all_blocks:
        if b.get("type") == DOC_TITLE_TYPE:
            documenttitle_text = _block_text(b) or documenttitle_text
            if documenttitle_text:
                break
    documenttitle_text = (documenttitle_text or "").strip()
    fallback_title = (doc_title or "").strip()
    if documenttitle_text and not is_garbled_text(documenttitle_text):
        title_text = documenttitle_text
    else:
        title_text = fallback_title
    if title_text and not is_garbled_text(title_text):
        title_chunk = _build_doc_title_chunk(title_text, page=0)
        if title_chunk:
            chunks.append(title_chunk)

    # 状态
    cur_section = ""
    cur_section_is_summary = False
    cur_section_is_references = False
    pending_ref_blocks: List[Dict[str, Any]] = []  # 累积 reference 块 (引用条目)
    # 累积连续 paragraph 块, 在被非段落块 (title/公式/表/图/reference) 打断时,
    # 用 MinerU 同款逻辑段落分组 (误切合并 + 短段合并 + 期刊元数据行丢弃) 后再成 chunk.
    pending_para_blocks: List[Dict[str, Any]] = []
    paragraph_counter = [0]

    def _next_para_idx() -> int:
        paragraph_counter[0] += 1
        return paragraph_counter[0]

    def _flush_pending_references() -> None:
        nonlocal pending_ref_blocks
        if not pending_ref_blocks:
            return
        ref_chunks = _build_references_chunks_from_blocks(
            pending_ref_blocks,
            section=cur_section if cur_section_is_references else "References",
            batch_size=references_batch_size,
        )
        chunks.extend(ref_chunks)
        pending_ref_blocks = []

    # 摘要 section 内所有 paragraph 累积起来, 整段合成单个 summary chunk (避免被段落切碎)
    pending_summary_texts: List[str] = []
    pending_summary_pages: set = set()

    def _flush_pending_summary() -> None:
        nonlocal pending_summary_texts, pending_summary_pages
        if not pending_summary_texts:
            return
        content = "\n\n".join(t for t in pending_summary_texts if t.strip()).strip()
        content = _clip_summary_text(_strip_summary_prefix(content))
        if content:
            chunks.append({
                "id": _short_id("summary"),
                "type": "summary",
                "section": cur_section or "",
                "pages": sorted(pending_summary_pages),
                "content": content,
                "context": "",
                "related_assets": [],
                "paragraph_index": _next_para_idx(),
            })
        pending_summary_texts = []
        pending_summary_pages = set()

    def _flush_pending_paragraphs() -> None:
        """把缓冲的连续 paragraph 块按 MinerU 逻辑段落分组后成 text chunk。

        复用 ``_group_logical_paragraphs`` (= 误切合并 + 短段合并 + 期刊元数据行
        丢弃), 与 MinerU 支路同款; 之后长段仍走 ``_maybe_split_text_chunk``.
        preamble (首个 section title 之前的正文) 不占段号, 打 ``is_preamble``,
        与 MinerU 对齐。
        """
        nonlocal pending_para_blocks
        if not pending_para_blocks:
            return
        blocks = pending_para_blocks
        pending_para_blocks = []
        # 转成 MinerU content_list_v2 风格的 paragraph item, 复用其分组逻辑;
        # 额外挂 _page 以便分组后还原页码 (分组函数不读该字段, 不受影响)。
        mineru_items: List[Dict[str, Any]] = []
        for pb in blocks:
            txt = _block_text(pb)
            if not txt:
                continue
            mineru_items.append({
                "type": "paragraph",
                "content": {"paragraph_content": [{"type": "text", "content": txt}]},
                "_page": int(pb.get("page", 0)),
            })
        if not mineru_items:
            return
        groups = _group_logical_paragraphs(mineru_items)
        is_preamble = (not cur_section.strip()) and not cur_section_is_summary
        for group in groups:
            combined = _group_combined_text(group)
            if not combined:
                continue
            pages_in_group = sorted({int(it.get("_page", 0)) for it in group})
            page = pages_in_group[0] if pages_in_group else 0
            para_idx = -1 if is_preamble else _next_para_idx()
            base = _build_paragraph_chunk(
                combined, section=cur_section, page=page, paragraph_index=para_idx,
            )
            if not base:
                continue
            if pages_in_group:
                base["pages"] = pages_in_group
            if is_preamble:
                base["is_preamble"] = True
            split = _maybe_split_text_chunk(
                base, embedder,
                target_chars=split_target_chars,
                max_chars=split_max_chars,
                min_chars=split_min_chars,
                breakpoint_percentile=split_breakpoint_percentile,
                length_fn=split_length_fn,
                overlap_chars=split_overlap,
            )
            for sc in split:
                if "paragraph_index" not in sc:
                    sc["paragraph_index"] = base["paragraph_index"]
                if is_preamble and sc.get("type") == "text":
                    sc["is_preamble"] = True
            chunks.extend(split)

    # 主循环: 按全局阅读序遍历 blocks
    for idx, b in enumerate(all_blocks):
        t = b.get("type")

        # —— section 切换 ——
        if t == TITLE_TYPE:
            # 在切 section 之前, 清空当前 section 的累积 (正文 / references / summary)
            _flush_pending_paragraphs()
            _flush_pending_references()
            _flush_pending_summary()
            new_section = sanitize_section(_block_text(b))
            cur_section = new_section
            cur_section_is_summary = bool(
                new_section and new_section in summary_sections
            )
            cur_section_is_references = _is_references_section_text(new_section)
            continue

        if t == DOC_TITLE_TYPE:
            continue  # 已在前面注入了

        # —— references 章节: 累积 reference 块 ——
        if cur_section_is_references:
            # references section 里通常都是 reference 类块; paragraph 也兼容
            if t == REFERENCE_TYPE or t in TEXT_LIKE_TYPES:
                pending_ref_blocks.append(b)
            # 其它类型 (公式/图/表) 在 references section 里基本不会出现; 忽略
            continue

        # —— reference 类块在非 references section 里: 也按 references 类型聚合 ——
        # (UniParser 偶尔把零散引用判到正文段尾, 比如 last page 末尾, 没显式 title 切换)
        if t == REFERENCE_TYPE:
            _flush_pending_paragraphs()  # reference 打断正文段, 先 flush 已缓冲的段落
            pending_ref_blocks.append(b)
            continue
        # 一旦碰到非 reference 块, 先 flush 之前累积的 reference (如果有)
        if pending_ref_blocks:
            _flush_pending_references()

        # —— 摘要 section: paragraph 累积, 其它类型 (公式/图/表) 仍正常出 chunk ——
        if cur_section_is_summary and t in TEXT_LIKE_TYPES:
            txt = _block_text(b)
            # 期刊元数据行 (中图分类号/DOI/收稿日期 等) 不进摘要, 与 MinerU 对齐
            if txt and not _is_metadata_line(txt):
                pending_summary_texts.append(txt)
                pending_summary_pages.add(int(b.get("page", 0)))
            continue

        # —— equation 块 -> equation chunk ——
        if t == EQUATION_TYPE:
            _flush_pending_paragraphs()  # 公式打断正文段, 先把缓冲段落成 chunk
            _flush_pending_summary()  # 公式作为 section 内边界, summary 先 flush
            latex = (b.get("latex_repr") or "").strip()
            if not latex:
                continue
            eq_label_text = _equation_caption_for(b, idx, all_blocks)
            ec = _build_equation_chunk(
                latex, section=cur_section,
                page=int(b.get("page", 0)),
                eq_label=eq_label_text,
            )
            if ec:
                chunks.append(ec)
            continue

        # —— table 块 -> table chunk ——
        if t == TABLE_TYPE:
            _flush_pending_paragraphs()
            _flush_pending_summary()
            # source=pages_tree 时按 group_id 去重
            gid = b.get("_group_id")
            if gid is not None and gid in seen_table_groups:
                continue
            cap, lab = _table_caption_for(b, idx, all_blocks)
            tc = _build_table_chunk(b, cur_section, cap, lab)
            if tc:
                chunks.append(tc)
                if lab:
                    table_idx.setdefault(lab, tc["id"])
                if gid is not None:
                    seen_table_groups.add(gid)
            continue

        # —— imagecaption -> image chunk (锚定 1 个 logical figure, 不论它含几个子面板) ——
        if t == IMAGECAPTION_TYPE:
            _flush_pending_paragraphs()
            _flush_pending_summary()
            # source=pages_tree 时按 group_id 去重: 同 group 多个 caption 只取一次
            gid = b.get("_group_id")
            if gid is not None and gid in seen_image_groups:
                continue
            ic = _build_image_chunk_from_caption(b, cur_section)
            if ic:
                chunks.append(ic)
                lab = ic.get("_label")
                if lab:
                    figure_idx.setdefault(lab, ic["id"])
                if gid is not None:
                    seen_image_groups.add(gid)
            continue

        # —— 视觉块 (figure/image/chart) 与子面板标签 (legend) 都不出 chunk: ——
        # 它们的语义已经由对应的 imagecaption 承载; 单独保留只会让 image
        # 类型被 1 篇文章塞进几十个空 caption (e.g. "(a)") 噪音
        if t in FIGURE_VISUAL_TYPES or t == LEGEND_TYPE:
            continue

        # —— 其它 caption 类 (tablecaption / equationid): 已经被 table / equation
        # 锚点循环就近消费, 主循环里直接跳 ——
        if t in CAPTION_TYPES:
            continue

        # —— paragraph: 先缓冲, 在被非段落块打断 / section 结束时统一做逻辑段落分组 ——
        # (对齐 MinerU: 误切合并 + 短段合并 + 期刊元数据行丢弃, 再走 semantic_split)
        if t in TEXT_LIKE_TYPES:
            if _block_text(b):
                pending_para_blocks.append(b)
            continue

        # —— 其它未知类型: 默默丢 (有 text 字段才记录 warning) ——
        if _block_text(b):
            logger.debug(f"[uniparser-chunker] 未处理 type={t!r} page={b.get('page')}")

    # —— 收尾 flush ——
    _flush_pending_paragraphs()
    _flush_pending_summary()
    _flush_pending_references()

    # —— v6: content-driven summary promote (LLM 兜底之前先跑) ——
    # 处理用户反馈的 case: section="" 但 content 段首 "摘要：..." 的 text chunk
    if summary_enabled:
        _promote_text_to_summary(
            chunks,
            text_patterns=summary_text_patterns,
            stop_patterns=summary_stop_patterns,
            max_promote=int(summary_max_sections),
        )

    # —— tier 4: LLM 兜底摘要 (与 MinerU 同策略, 受 summary_llm_enabled 严格控制) ——
    has_summary = any(c.get("type") == "summary" for c in chunks)
    if has_summary:
        pass  # tier1/1.5/2/3 已产出 summary, 不需要兜底
    elif not summary_llm_enabled:
        logger.info(
            f"[uniparser-chunker][summary] tier1-3 + 段首 promote 均未命中 "
            f"(section_strategy={detection.get('strategy')!r}); "
            f"chunking.summary.llm.enabled=false, 不调 LLM, 该文档无 summary chunk"
        )
    elif llm is None:
        logger.warning(
            f"[uniparser-chunker][summary] tier1-3 + 段首 promote 均未命中 "
            f"(section_strategy={detection.get('strategy')!r}), 未提供 LLMClient "
            f"(检查 generation.api_key), 该文档无 summary chunk"
        )
    else:
        first_page_text = _collect_first_page_text_uniparser(
            all_blocks, max_chars=summary_llm_max_input_chars,
        )
        llm_summary = _llm_synthesize_summary(
            first_page_text,
            llm,
            temperature=summary_llm_temperature,
            max_tokens=summary_llm_max_tokens,
            disable_thinking=summary_llm_disable_thinking,
            system_prompt=summary_llm_system_prompt,
            user_prompt_template=summary_llm_user_template,
        )
        if llm_summary:
            logger.info(
                f"[uniparser-chunker][summary][tier4] LLM 兜底生成 summary chunk "
                f"(strategy={detection.get('strategy')!r})"
            )
            # 插到 title chunk 后面 (如果有), 否则插到最前
            summary_chunk = {
                "id": _short_id("summary"),
                "type": "summary",
                "section": "[LLM-synthesized abstract]",
                "pages": [0],
                "content": llm_summary,
                "context": "",
                "related_assets": [],
                "paragraph_index": 0,
            }
            insert_at = 0
            for i, c in enumerate(chunks):
                if c.get("type") == "title":
                    insert_at = i + 1
                    break
            chunks.insert(insert_at, summary_chunk)

    # —— 章节锚点兜底 (对齐 MinerU chunker._flush_section 的 section 锚点链接): ——
    # 同一 section 内, 每个 text/summary chunk 挂上本节所有 image/table chunk,
    # asset chunk 反向挂回本节首个 text/summary chunk。这样即便正文没写 "图N/Fig.N",
    # 检索侧 assets 邻域扩展也能把本节图表作为互补上下文带出 (修复"问图表找不到")。
    # 必须在 cross-ref 之前跑 (此时 asset 的 _label 还在, 用于生成可读 label)。
    _link_section_assets(chunks)

    # —— cross-ref: 扫 Fig. N / Table N 字面量, **合并**进 related_assets ——
    # 合并而非覆盖: 保留上面 _link_section_assets 建立的章节锚点链接, 再并入
    # 显式交叉引用 (按 chunk_id 去重), 与 MinerU chunker 的合并语义一致。
    for c in chunks:
        if c.get("type") == "references":
            c.pop("_label", None)
            continue
        ref_text = (c.get("content") or "" if c["type"] in ("text", "summary")
                    else (c.get("content") or "") + " " + (c.get("context") or ""))
        refs = _scan_cross_refs(ref_text)
        related: List[Dict[str, str]] = list(c.get("related_assets") or [])
        seen_ids = {a.get("chunk_id") for a in related if isinstance(a, dict)}
        self_label = c.get("_label")
        for fig in refs.get("figures", []):
            if fig == self_label:
                continue
            fid = figure_idx.get(fig)
            if fid and fid != c["id"] and fid not in seen_ids:
                related.append({"type": "image", "label": f"Fig. {fig}", "chunk_id": fid})
                seen_ids.add(fid)
        for tab in refs.get("tables", []):
            if tab == self_label:
                continue
            tid = table_idx.get(tab)
            if tid and tid != c["id"] and tid not in seen_ids:
                related.append({"type": "table", "label": f"Table {tab}", "chunk_id": tid})
                seen_ids.add(tid)
        c["related_assets"] = related
        c.pop("_label", None)

    # —— equation 双向 link (与 MinerU 同函数) ——
    _link_equations_to_text(chunks)

    return chunks


# ── 工具函数 ───────────────────────────────────────────────────────────


def _link_section_assets(chunks: List[Dict[str, Any]]) -> None:
    """章节锚点兜底: 按文档阅读序的连续 section, 双向关联本节 image/table 与
    text/summary chunk (对齐 MinerU ``chunker._flush_section`` 的 section 锚点)。

    - 每个 text/summary chunk.related_assets += 本节每个 asset (合并去重);
    - 每个 asset chunk.related_assets += 本节首个 text/summary chunk (anchor)。

    references chunk 不参与 (被引文献的图号不应链到本文图表), 同时作为分段边界。
    合并而非覆盖: 由调用方后续的 cross-ref 再并入显式 "图N/表N" 引用。
    asset 的 label 取其 ``_label`` (Fig./Table 编号); 没有编号则退化为类型名。
    """
    asset_types = ("image", "table")
    text_types = ("text", "summary")
    n = len(chunks)
    i = 0
    while i < n:
        if chunks[i].get("type") == "references":
            i += 1
            continue
        sec = chunks[i].get("section")
        # 收集连续同 section 的一段 (遇 references 或 section 变化即断开)
        j = i
        run: List[Dict[str, Any]] = []
        while j < n:
            cj = chunks[j]
            if cj.get("type") == "references" or cj.get("section") != sec:
                break
            run.append(cj)
            j += 1

        texts = [x for x in run if x.get("type") in text_types]
        assets = [x for x in run if x.get("type") in asset_types]
        if texts and assets:
            # text/summary ← 本节所有 asset
            for tc in texts:
                rel = list(tc.get("related_assets") or [])
                seen = {a.get("chunk_id") for a in rel if isinstance(a, dict)}
                for ac in assets:
                    aid = ac.get("id")
                    if not aid or aid in seen:
                        continue
                    lab = ac.get("_label")
                    if ac.get("type") == "image":
                        label = f"Fig. {lab}" if lab else "image"
                    else:
                        label = f"Table {lab}" if lab else "table"
                    rel.append({"type": ac["type"], "label": label, "chunk_id": aid})
                    seen.add(aid)
                tc["related_assets"] = rel
            # asset → 本节首个 text/summary (anchor)
            anchor = texts[0]
            anchor_id = anchor.get("id")
            for ac in assets:
                rel = list(ac.get("related_assets") or [])
                seen = {a.get("chunk_id") for a in rel if isinstance(a, dict)}
                if anchor_id and anchor_id not in seen:
                    rel.append({
                        "type": anchor["type"],
                        "label": sec or "section_text",
                        "chunk_id": anchor_id,
                    })
                ac["related_assets"] = rel

        i = j if j > i else i + 1


def _is_references_section_text(section: str) -> bool:
    """与 chunker._is_references_section 同义; 这里独立一份避免 import 循环风险."""
    if not section:
        return False
    return any(p.match(section) for p in REFERENCES_SECTION_PATTERNS)


def write_meta_sidecar(
    blocks_output_path: str,
    result_json: Dict[str, Any],
    doc_id: Optional[str] = None,
    doc_name: Optional[str] = None,
    publication_year: Optional[int] = None,
) -> str:
    """在 ``knowledge_blocks.json`` 旁写 ``knowledge_blocks_meta.json``,
    让 ``MilvusIngester._load_meta_sidecar`` 在 ingest 阶段自动吃到.

    Returns:
        写入的 meta 路径
    """
    base, _ = os.path.splitext(blocks_output_path)
    # _load_meta_sidecar 找的是 <base>_meta.json (在 _vec 后缀剥掉之后)
    meta_path = base + "_meta.json"
    meta: Dict[str, Any] = {
        "source": "uniparser",
        "token": result_json.get("token"),
        "filename": result_json.get("filename"),
        "lang": result_json.get("lang"),
        "total_page": result_json.get("total_page"),
        "total_textual": result_json.get("total_textual"),
        "total_table": result_json.get("total_table"),
        "total_equa": result_json.get("total_equa"),
        "total_figure": result_json.get("total_figure"),
        "total_chart": result_json.get("total_chart"),
        "total_mol": result_json.get("total_mol"),
    }
    if doc_id:
        meta["doc_id"] = doc_id
    if doc_name:
        meta["doc_name"] = doc_name
    if publication_year:
        meta["publication_year"] = int(publication_year)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


def load_uniparser_result(path: str) -> Dict[str, Any]:
    """加载 uniparser_result.json, 不做 schema 校验 (上层 build_knowledge_blocks_uniparser
    会检查 pages_dict)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def autodiscover_uniparser_result(root_dir: str) -> Optional[str]:
    """在 root_dir 里 (递归) 找最新的 uniparser_result.json."""
    import glob
    matches = sorted(
        glob.glob(os.path.join(root_dir, "**", "uniparser_result.json"), recursive=True),
        key=lambda p: os.path.getmtime(p),
    )
    return matches[-1] if matches else None
