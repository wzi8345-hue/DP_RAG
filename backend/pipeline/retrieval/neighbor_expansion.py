"""邻域扩展检索 (依赖图谱场景 #3 邻域探索 / #4 跨模态 / #6 同义扩展)。

把"从一个 chunk 节点沿边走到相关 chunk 节点"做成一个**后处理算子**: 给定一组已召回
的种子 hit, 沿以下几类边各扩 1 跳, 回填到检索结果里 (不替换种子):

  - assets   : 沿 chunk.related_assets[].chunk_id 走 (图/表/公式 ↔ 正文的交叉引用边,
               ingest 阶段已建好) —— 跨模态 / 邻域
  - adjacent : 同 doc_id + 相邻 paragraph_index (±window) —— "图N附近的文字" / 邻域探索
  - page     : 同 doc_id + 相邻 page_start (±window) —— 版面邻近
  - similar  : more-like-this, 用种子内容做向量近邻 (可跨文献) —— 同义/同类扩展

设计原则 (对齐 karpathy 指南: 最小、外科手术式):
  1. 纯后处理, 不改任何现有检索器内部逻辑; 不触发时零行为变化;
  2. 复用现有 Milvus client / VectorRetriever / Hit / _row_to_hit, 不新建连接;
  3. 永不抛: 任一边查询失败只 warning 并跳过, 返回已得到的邻居;
  4. 去重 (对种子 + 邻居自身), 总量封顶, 避免 context 膨胀。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .retrievers import Hit, _escape_eq, _row_to_hit, _OUTPUT_FIELDS

logger = logging.getLogger(__name__)

# 扩展模式常量 (与 RouteDecision.expand_neighbors / FC schema expand 字段一致)
EXPAND_ASSETS = "assets"
EXPAND_ADJACENT = "adjacent"
EXPAND_PAGE = "page"
EXPAND_SIMILAR = "similar"
VALID_EXPAND_MODES = frozenset(
    {EXPAND_ASSETS, EXPAND_ADJACENT, EXPAND_PAGE, EXPAND_SIMILAR}
)

# 邻域扩展结果在 route_results 里的伪路由键 (下游 reranker / context_builder 按 list 泛化处理)
ROUTE_NEIGHBOR = "neighbor"

# 默认窗口 / 上限
_DEFAULT_ADJACENT_WINDOW = 1     # 相邻段落 ±1
_DEFAULT_PAGE_WINDOW = 0         # 同页 (0=仅本页)
_DEFAULT_SIMILAR_TOP_K = 5
_DEFAULT_MAX_TOTAL = 12          # 单轮邻域扩展回填的最大 chunk 数
_DEFAULT_QUERY_LIMIT = 200       # 单次 Milvus query 候选上限
_SIMILAR_SEED_CHARS = 500        # similar 模式拿种子内容做 query 的截断长度


def normalize_expand_modes(raw: Any) -> List[str]:
    """把任意输入清洗成合法的 expand 模式列表 (保序去重, 丢弃非法值)。"""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    for x in raw:
        m = str(x).strip().lower()
        if m in VALID_EXPAND_MODES and m not in out:
            out.append(m)
    return out


def _hit_key(hit: Hit) -> str:
    return hit.pk or hit.chunk_id or ""


def collect_seed_hits(route_results: Dict[str, Any]) -> List[Hit]:
    """从 route_results 抽出所有种子 hit (跳过已有的 neighbor 伪路由)。"""
    seeds: List[Hit] = []
    for route, val in (route_results or {}).items():
        if route == ROUTE_NEIGHBOR:
            continue
        chunk_hits = getattr(val, "chunk_hits", None)
        if chunk_hits is not None:
            seeds.extend(h for h in chunk_hits if isinstance(h, Hit))
        elif isinstance(val, list):
            seeds.extend(h for h in val if isinstance(h, Hit))
    return seeds


class NeighborExpander:
    """沿 chunk 间的边做 1 跳邻域扩展。

    Args:
        client: 复用的 Milvus client (与各 retriever 共享同一连接)。
        collection: collection 名。
        vector_retriever: 可选; 提供时才支持 similar (more-like-this) 模式。
        adjacent_window / page_window: 段落 / 页码扩展窗口。
        similar_top_k: similar 模式每个种子取的近邻数。
        max_total: 单次扩展回填上限。
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        *,
        vector_retriever: Any = None,
        adjacent_window: int = _DEFAULT_ADJACENT_WINDOW,
        page_window: int = _DEFAULT_PAGE_WINDOW,
        similar_top_k: int = _DEFAULT_SIMILAR_TOP_K,
        max_total: int = _DEFAULT_MAX_TOTAL,
        query_limit: int = _DEFAULT_QUERY_LIMIT,
    ) -> None:
        self.client = client
        self.collection = collection
        self.vec = vector_retriever
        self.adjacent_window = max(0, int(adjacent_window))
        self.page_window = max(0, int(page_window))
        self.similar_top_k = max(1, int(similar_top_k))
        self.max_total = max(1, int(max_total))
        self.query_limit = max(1, int(query_limit))

    # ── 公共 API ──────────────────────────────────────────────────────────

    def expand(
        self,
        seed_hits: Sequence[Hit],
        modes: Sequence[str],
        *,
        time_filter: Optional[str] = None,
        max_total: Optional[int] = None,
    ) -> List[Hit]:
        """对种子 hit 沿 modes 指定的边做 1 跳扩展, 返回去重 + 封顶后的邻居 hit。"""
        modes = normalize_expand_modes(modes)
        seeds = [h for h in (seed_hits or []) if isinstance(h, Hit)]
        if not modes or not seeds:
            return []

        cap = self.max_total if max_total is None else max(1, int(max_total))
        seed_keys = {_hit_key(h) for h in seeds if _hit_key(h)}

        collected: List[Hit] = []
        seen: set = set(seed_keys)

        for mode in modes:
            if mode == EXPAND_ASSETS:
                cand = self._expand_assets(seeds, time_filter)
            elif mode == EXPAND_ADJACENT:
                cand = self._expand_adjacent(seeds, time_filter)
            elif mode == EXPAND_PAGE:
                cand = self._expand_page(seeds, time_filter)
            elif mode == EXPAND_SIMILAR:
                cand = self._expand_similar(seeds, time_filter)
            else:
                cand = []

            for hit in cand:
                key = _hit_key(hit)
                if not key or key in seen:
                    continue
                seen.add(key)
                hit.sources = [f"{ROUTE_NEIGHBOR}:{mode}"]
                hit.stage = ROUTE_NEIGHBOR
                collected.append(hit)
                if len(collected) >= cap:
                    logger.info(
                        f"[neighbor] 达到上限 {cap}, 截断 (modes={modes})"
                    )
                    return collected
        logger.info(
            f"[neighbor] modes={modes} seeds={len(seeds)} -> 扩展 {len(collected)} chunk"
        )
        return collected

    # ── 各模式实现 ────────────────────────────────────────────────────────

    def _expand_assets(
        self, seeds: Sequence[Hit], time_filter: Optional[str],
    ) -> List[Hit]:
        """沿 related_assets[].chunk_id 走 (图/表/公式 ↔ 正文交叉引用边)。"""
        chunk_ids: List[str] = []
        seen: set = set()
        for h in seeds:
            for a in (h.related_assets or []):
                if not isinstance(a, dict):
                    continue
                cid = str(a.get("chunk_id") or "").strip()
                if cid and cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        if not chunk_ids:
            return []
        in_list = ", ".join(f'"{_escape_eq(c)}"' for c in chunk_ids)
        clause = f"chunk_id in [{in_list}]"
        return self._query(_and(clause, time_filter))

    def _expand_adjacent(
        self, seeds: Sequence[Hit], time_filter: Optional[str],
    ) -> List[Hit]:
        """同 doc_id + 相邻 paragraph_index (±window)。"""
        w = self.adjacent_window
        per_doc: Dict[str, set] = {}
        for h in seeds:
            if not h.doc_id or h.paragraph_index is None or h.paragraph_index < 1:
                continue
            targets = per_doc.setdefault(h.doc_id, set())
            for d in range(-w, w + 1):
                idx = h.paragraph_index + d
                if idx >= 1:
                    targets.add(idx)
        out: List[Hit] = []
        for doc_id, idxs in per_doc.items():
            if not idxs:
                continue
            in_list = ", ".join(str(i) for i in sorted(idxs))
            clause = f'doc_id == "{_escape_eq(doc_id)}" and paragraph_index in [{in_list}]'
            out.extend(self._query(_and(clause, time_filter)))
        return out

    def _expand_page(
        self, seeds: Sequence[Hit], time_filter: Optional[str],
    ) -> List[Hit]:
        """同 doc_id + 相邻 page_start (±window); 默认仅本页。"""
        w = self.page_window
        per_doc: Dict[str, set] = {}
        for h in seeds:
            if not h.doc_id or h.page_start is None or h.page_start < 0:
                continue
            targets = per_doc.setdefault(h.doc_id, set())
            for d in range(-w, w + 1):
                p = h.page_start + d
                if p >= 0:
                    targets.add(p)
        out: List[Hit] = []
        for doc_id, pages in per_doc.items():
            if not pages:
                continue
            in_list = ", ".join(str(p) for p in sorted(pages))
            clause = f'doc_id == "{_escape_eq(doc_id)}" and page_start in [{in_list}]'
            out.extend(self._query(_and(clause, time_filter)))
        return out

    def _expand_similar(
        self, seeds: Sequence[Hit], time_filter: Optional[str],
    ) -> List[Hit]:
        """more-like-this: 用种子内容做向量近邻 (可跨文献)。需注入 vector_retriever。"""
        if self.vec is None:
            logger.info("[neighbor] similar 模式无 vector_retriever, 跳过")
            return []
        # 仅用得分最高的种子 (route_results 通常已按相关性给出靠前的种子) 做 anchor,
        # 避免多种子 query 噪音累积。
        anchor = seeds[0]
        seed_text = (anchor.content or "").strip()[:_SIMILAR_SEED_CHARS]
        if not seed_text:
            return []
        try:
            hits = self.vec.retrieve(
                seed_text, top_k=self.similar_top_k, filter_expr=time_filter or None,
            )
        except Exception as e:
            logger.warning(f"[neighbor] similar 向量近邻失败: {e}")
            return []
        return hits or []

    # ── Milvus query 封装 ─────────────────────────────────────────────────

    def _query(self, filter_expr: str) -> List[Hit]:
        if not filter_expr:
            return []
        try:
            rows = self.client.query(
                collection_name=self.collection,
                filter=filter_expr,
                output_fields=_OUTPUT_FIELDS,
                limit=self.query_limit,
            )
        except Exception as e:
            logger.warning(f"[neighbor] query 失败 filter={filter_expr[:120]!r}: {e}")
            return []
        return [_row_to_hit(row) for row in (rows or [])]


def _and(clause: str, time_filter: Optional[str]) -> str:
    """把主 clause 与可选 time_filter 用 and 连接。"""
    if clause and time_filter:
        return f"({clause}) and ({time_filter})"
    return clause or (time_filter or "")


# ---------------------------------------------------------------------------
# 编排辅助: 把邻域扩展接到 route_results 上 (供 _dispatch / retrieve_node 复用)
# ---------------------------------------------------------------------------

def collect_expand_modes(decisions: Sequence[Any]) -> List[str]:
    """从一批 RouteDecision (单 / multi 的各 sub) 合并出去重的 expand 模式列表。"""
    modes: List[str] = []
    for d in decisions or []:
        for m in normalize_expand_modes(getattr(d, "expand_neighbors", None)):
            if m not in modes:
                modes.append(m)
    return modes


def apply_neighbor_expansion(
    route_results: Dict[str, Any],
    *,
    modes: Sequence[str],
    expander: Optional[NeighborExpander],
    time_filter: Optional[str] = None,
    max_total: Optional[int] = None,
    cid: str = "-",
) -> Dict[str, Any]:
    """对 route_results 的种子 hit 做邻域扩展, 把邻居回填到 results["neighbor"]。

    不触发 (无 modes / 无 expander / 无种子 / 无邻居) 时**原样返回** route_results,
    保证零行为变化。
    """
    modes = normalize_expand_modes(modes)
    if not modes or expander is None:
        return route_results
    seeds = collect_seed_hits(route_results)
    if not seeds:
        logger.info(f"[{cid}] [neighbor] 无种子 hit, 跳过扩展 (modes={modes})")
        return route_results
    neighbors = expander.expand(
        seeds, modes, time_filter=time_filter, max_total=max_total,
    )
    if not neighbors:
        return route_results

    merged = dict(route_results)
    existing = merged.get(ROUTE_NEIGHBOR)
    if isinstance(existing, list):
        keys = {_hit_key(h) for h in existing if isinstance(h, Hit)}
        merged[ROUTE_NEIGHBOR] = existing + [
            h for h in neighbors if _hit_key(h) not in keys
        ]
    else:
        merged[ROUTE_NEIGHBOR] = neighbors
    logger.info(
        f"[{cid}] [neighbor] 回填 {len(neighbors)} 个邻居 chunk 到 results['neighbor'] "
        f"(modes={modes})"
    )
    return merged
