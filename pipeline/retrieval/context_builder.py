"""上下文构建器: 把 Hits 拼成 LLM-ready 上下文文本。

从原始 retrieve.py 的 ContextBuilder 搬入。
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..processors.chunker import sanitize_section
from ..retrieval.retrievers import Hit

logger = logging.getLogger(__name__)


class ContextBuilder:
    """把 Hits 拼成 LLM-ready 上下文文本 (v2 schema)。

    格式约定:
    - 每个 chunk 是一段, 用 "---" 分隔
    - 段头标注 source / score / chunk 类型 / 章节 / 页码 / 年份 / chunk_id
    - 主体复用 chunk.content
    - context 和 related_assets 单独列出
    """

    SEP = "\n\n---\n\n"

    def build(self, hits: List[Hit], query: Optional[str] = None) -> str:
        if not hits:
            return "[No relevant context retrieved]"

        sections: List[str] = []
        if query:
            sections.append(f"# Query\n{query}")
        sections.append(f"# Retrieved Context ({len(hits)} chunks)")

        for i, hit in enumerate(hits, 1):
            sections.append(self._format_hit(i, hit))
        return self.SEP.join(sections)

    def _format_hit(self, rank: int, hit: Hit) -> str:
        lines: List[str] = []
        score_parts = []
        if hit.rrf_score:
            score_parts.append(f"rrf={hit.rrf_score:.4f}")
        if hit.score:
            score_parts.append(f"raw={hit.score:.3f}")
        score_str = ", ".join(score_parts) or "n/a"
        src_str = "+".join(hit.sources) if hit.sources else "?"

        head_bits = [
            f"## [{rank}] {hit.type.upper()}",
            f"source={src_str}",
            score_str,
            f"doc={hit.doc_id}",
            f"page={hit.page_start}",
            f"chunk_id={hit.chunk_id}",
        ]
        if hit.publication_year:
            head_bits.append(f"year={hit.publication_year}")
        lines.append(" | ".join(head_bits))

        clean_section = sanitize_section(hit.section)
        if clean_section:
            lines.append(f"**Section:** {clean_section}")
        if hit.matched_keywords:
            lines.append(f"**Matched:** {', '.join(hit.matched_keywords)}")

        if hit.content:
            lines.append("")
            lines.append(hit.content)

        if hit.context:
            ctx = hit.context
            lines.append("")
            lines.append(f"**Context:** {ctx}")

        if hit.related_assets:
            related_bits = []
            for r in hit.related_assets:
                if not isinstance(r, dict):
                    continue
                label = r.get("label") or "?"
                cid = r.get("chunk_id") or "?"
                related_bits.append(f"{label} (chunk_id={cid})")
            if related_bits:
                lines.append(f"**Related:** {' ; '.join(related_bits)}")
        return "\n".join(lines)
