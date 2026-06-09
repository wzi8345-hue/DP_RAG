"""企业内部材料数据库检索源 (占位 / 预留)。

规划: 自然语言 → text2sql → 受控只读 SQL → Postgres 查询 → 结构化 Hit。
当前未实现; 前端将该源标记为 Beta/灰显。实现此类并 register_source 即可挂载,
无需改动问答主流程 (由 RetrievalSource 抽象隔离)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..auth import AuthContext


class EnterpriseSqlSource:
    key = "enterprise_sql"

    def __init__(self, dsn: str = "") -> None:
        self._dsn = dsn

    def retrieve(
        self,
        query: str,  # noqa: ARG002
        *,
        ctx: AuthContext,  # noqa: ARG002
        params: dict[str, Any],  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("enterprise_sql 检索源尚未实现 (见 DEV_PLAN M7)")

    def health(self) -> str:
        return "disabled"
