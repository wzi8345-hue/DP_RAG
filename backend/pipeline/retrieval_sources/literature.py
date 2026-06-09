"""文献库检索源: 封装现有 pipeline 的 Milvus 检索。

当前 chat/query 直接走 pipeline; 本适配器是 M7 统一多源融合的落点。
集成时由 QueryFlow 调用 retrieve(), 把 collection/doc_ids/top_k/mode 透传给 pipeline。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..auth import AuthContext
    from ..pipeline import Pipeline


class LiteratureSource:
    key = "literature"

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipe = pipeline

    def retrieve(
        self,
        query: str,
        *,
        ctx: AuthContext,  # noqa: ARG002  归属过滤在 M5 接入
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._pipe.query(
            query=query,
            mode=params.get("mode"),
            top_k=params.get("top_k"),
            use_agentic=params.get("use_agentic", True),
            professional=params.get("professional", False),
            collection=params.get("collection"),
        )
        return list(getattr(result, "hits", []) or [])

    def health(self) -> str:
        try:
            self._pipe.stats()
            return "ok"
        except Exception as e:  # pragma: no cover
            return f"error: {e}"
