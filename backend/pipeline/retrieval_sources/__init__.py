"""多检索源抽象 (预留企业 SQL 接入)。

当前仅 literature 落地; enterprise_sql 留接口与开关 (前端 Beta/灰显)。
查询请求体的 sources[] 决定启用哪些源, 后端按 AuthContext 鉴权后并行检索并融合。

集成点 (M7, 待落地):
- QueryFlow / chat 在检索阶段根据 req.sources 选择已注册的 RetrievalSource;
- 每个源返回统一的 Hit 列表, 由 context_builder 融合。
"""

from .base import RetrievalSource, get_registry, register_source

__all__ = ["RetrievalSource", "get_registry", "register_source"]
