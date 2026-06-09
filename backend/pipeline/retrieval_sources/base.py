"""检索源协议 + 注册表。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..auth import AuthContext


@runtime_checkable
class RetrievalSource(Protocol):
    """统一检索源接口。新增数据源 (如企业 SQL) 实现此协议并 register_source 即可挂载。"""

    key: str  # "literature" | "enterprise_sql" | ...

    def retrieve(
        self,
        query: str,
        *,
        ctx: AuthContext,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """返回统一结构的 Hit 列表 (dict)。params 含 collection/doc_ids/top_k/mode 等。"""
        ...

    def health(self) -> str:
        """返回该源的健康状态字符串。"""
        ...


_REGISTRY: dict[str, RetrievalSource] = {}


def register_source(source: RetrievalSource) -> None:
    _REGISTRY[source.key] = source


def get_registry() -> dict[str, RetrievalSource]:
    return _REGISTRY
