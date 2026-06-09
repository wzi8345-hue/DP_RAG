"""检索模块包: 提供向量检索、元数据检索、混合检索和 Agentic RAG。"""

from .retrievers import (
    MetadataRetriever,
    VectorRetriever,
    HybridRetriever,
    parse_query,
    ParsedQuery,
    build_retrievers,
)
from .context_builder import ContextBuilder
from .agentic import (
    QueryRouter,
    SummaryRetriever,
    ProgressiveLocalRetriever,
    EnhancedMetadataRetriever,
    AgenticContextBuilder,
    AgenticRAGPipeline,
    build_agentic_pipeline,
)

__all__ = [
    "MetadataRetriever",
    "VectorRetriever",
    "HybridRetriever",
    "parse_query",
    "ParsedQuery",
    "build_retrievers",
    "ContextBuilder",
    "QueryRouter",
    "SummaryRetriever",
    "ProgressiveLocalRetriever",
    "EnhancedMetadataRetriever",
    "AgenticContextBuilder",
    "AgenticRAGPipeline",
    "build_agentic_pipeline",
]

# LangGraph agent (可选依赖, langgraph 未安装时静默跳过)
try:
    from .langgraph_agent import (
        LangGraphAgent,
        build_langgraph_agent,
        build_langgraph_agent_from_pipeline,
    )
    __all__.extend([
        "LangGraphAgent",
        "build_langgraph_agent",
        "build_langgraph_agent_from_pipeline",
    ])
except ImportError:
    pass
