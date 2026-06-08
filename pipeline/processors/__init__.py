"""数据处理模块: 分块 (chunker) 和向量化 (vectorizer)。"""

from .chunker import build_knowledge_blocks, autodiscover_content_list_v2
from .uniparser_chunker import (
    build_knowledge_blocks_uniparser,
    autodiscover_uniparser_result,
    write_meta_sidecar as write_uniparser_meta_sidecar,
)
from .vectorizer import compose_embedding_text, html_table_to_text, vectorize_chunks

__all__ = [
    "build_knowledge_blocks",
    "autodiscover_content_list_v2",
    "build_knowledge_blocks_uniparser",
    "autodiscover_uniparser_result",
    "write_uniparser_meta_sidecar",
    "compose_embedding_text",
    "html_table_to_text",
    "vectorize_chunks",
]
