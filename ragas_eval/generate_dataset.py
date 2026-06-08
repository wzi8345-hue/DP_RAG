"""数据集生成器: 从 Milvus 中的 chunk 自动生成 RAGAS 评估数据集。

使用 LLM 根据 chunk 内容生成 question + ground_truth 对,
构建符合 RAGAS 输入格式的评估数据集。

用法:
    python generate_dataset.py                           # 默认生成 20 条
    python generate_dataset.py --num 50                  # 生成 50 条
    python generate_dataset.py --output datasets/my.json # 指定输出路径
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 从 Milvus 采样 chunk
# ---------------------------------------------------------------------------

def sample_chunks(
    milvus_uri: str = "./milvus_lite.db",
    milvus_token: str = "",
    collection: str = "literature_chunks",
    num_samples: int = 20,
    chunk_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """从 Milvus 随机采样 chunk, 优先选择内容丰富的 text/summary 类型。"""
    from pymilvus import MilvusClient

    kwargs: Dict[str, Any] = {
        "uri": milvus_uri,
        "keepalive_time_ms": 300_000,
        "keepalive_timeout_ms": 60_000,
    }
    if milvus_token:
        kwargs["token"] = milvus_token
    client = MilvusClient(**kwargs)

    if not client.has_collection(collection):
        raise ValueError(f"集合不存在: {collection}")

    # 优先采样 text 类型 (内容更丰富, 更适合生成问题)
    target_types = chunk_types or ["text", "summary"]
    filter_expr = " or ".join(f'type == "{t}"' for t in target_types)

    output_fields = [
        "chunk_id", "doc_id", "doc_name", "type", "section",
        "page_start", "content", "context",
    ]
    rows = client.query(
        collection_name=collection,
        filter=f"({filter_expr})",
        output_fields=output_fields,
        limit=num_samples * 5,
    )

    if not rows:
        logger.warning("未找到符合条件的 chunk, 降级到全类型")
        rows = client.query(
            collection_name=collection,
            filter="",
            output_fields=output_fields,
            limit=num_samples * 5,
        )

    # 过滤掉内容太短的 chunk
    rows = [r for r in rows if len(r.get("content", "")) >= 100]

    # 随机采样
    random.shuffle(rows)
    return rows[:num_samples]


# ---------------------------------------------------------------------------
# 用 LLM 生成 question + ground_truth
# ---------------------------------------------------------------------------

QA_GEN_SYSTEM_PROMPT = """你是一名科研文献QA生成专家。根据提供的文献片段，生成一个具体的、可回答的问题和参考答案。

要求:
1. 问题必须能从提供的片段中找到答案, 不要生成需要外部知识的问题
2. 问题应具体明确, 避免过于宽泛 (如"这篇文章讲了什么")
3. 参考答案应准确、简洁, 直接从片段内容中提取
4. 同时提取2-3个与问题最相关的关键句作为 ground_contexts

严格输出JSON格式, 不要任何解释或markdown围栏:
{"question": "问题", "ground_truth": "参考答案", "ground_contexts": ["关键句1", "关键句2"]}"""


def generate_qa_pairs(
    chunks: List[Dict[str, Any]],
    llm_api_base: str,
    llm_model: str,
    llm_api_key: str,
) -> List[Dict[str, Any]]:
    """对每个 chunk 调用 LLM 生成 question + ground_truth 对。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.clients.llm import LLMClient

    llm = LLMClient(
        api_base=llm_api_base,
        model=llm_model,
        api_key=llm_api_key,
        timeout=60,
        max_retries=2,
    )

    results: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        doc_name = chunk.get("doc_name", "未知文献")
        section = chunk.get("section", "")

        user_msg = (
            f"文献来源: {doc_name}\n"
            f"章节: {section}\n"
            f"片段内容:\n{content[:2000]}"
        )

        logger.info(f"[{i+1}/{len(chunks)}] 生成QA: {doc_name[:40]}...")

        try:
            resp = llm.chat(
                system=QA_GEN_SYSTEM_PROMPT,
                user=user_msg,
                temperature=0.3,
                max_tokens=300,
                disable_thinking=True,
            )
            raw = resp.get("answer", "")

            # 解析 JSON
            import re
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                qa = json.loads(json_match.group())
                results.append({
                    "question": qa.get("question", ""),
                    "ground_truth": qa.get("ground_truth", ""),
                    "ground_contexts": qa.get("ground_contexts", []),
                    "source_doc": doc_name,
                    "source_chunk_id": chunk.get("chunk_id", ""),
                })
                logger.info(f"  ✓ {qa.get('question', '')[:50]}...")
            else:
                logger.warning(f"  ✗ JSON 解析失败: {raw[:100]}")
        except Exception as e:
            logger.error(f"  ✗ LLM 调用失败: {e}")

        # 简单限流, 避免触发 API rate limit
        time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="生成 RAGAS 评估数据集")
    parser.add_argument("--num", type=int, default=20, help="生成问题数量")
    parser.add_argument("--output", default="datasets/auto_generated.json", help="输出路径")
    parser.add_argument("--config", default="config.yaml", help="评估配置文件")
    parser.add_argument("--milvus-uri", help="覆盖 Milvus URI")
    parser.add_argument("--types", nargs="*", help="chunk 类型过滤 (默认 text summary)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 加载配置
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 加载 pipeline 配置获取 Milvus 连接信息
    pipeline_cfg_path = cfg.get("pipeline_config", "../pipeline/default_config.yaml")
    with open(pipeline_cfg_path, "r", encoding="utf-8") as f:
        p_cfg = yaml.safe_load(f) or {}

    milvus_uri = args.milvus_uri or p_cfg.get("milvus", {}).get("uri", "./milvus_lite.db")
    milvus_token = p_cfg.get("milvus", {}).get("token", "")
    collection = p_cfg.get("milvus", {}).get("collection", "literature_chunks")

    # LLM 配置 (用于生成 QA)
    ragas_llm_cfg = cfg.get("ragas_llm", {})
    gen_cfg = p_cfg.get("generation", {})
    llm_api_base = ragas_llm_cfg.get("api_base") or gen_cfg.get("api_base", "")
    llm_model = ragas_llm_cfg.get("model") or gen_cfg.get("model", "")
    llm_api_key = ragas_llm_cfg.get("api_key") or gen_cfg.get("api_key", "")

    # 1. 采样 chunk
    logger.info(f"从 Milvus 采样 {args.num} 个 chunk...")
    chunks = sample_chunks(
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        collection=collection,
        num_samples=args.num,
        chunk_types=args.types,
    )
    logger.info(f"采样到 {len(chunks)} 个 chunk")

    if not chunks:
        logger.error("未采样到任何 chunk, 请检查 Milvus 数据库")
        sys.exit(1)

    # 2. 生成 QA 对
    logger.info("开始生成 QA 对...")
    qa_pairs = generate_qa_pairs(chunks, llm_api_base, llm_model, llm_api_key)
    logger.info(f"成功生成 {len(qa_pairs)} 个 QA 对")

    if not qa_pairs:
        logger.error("未生成任何 QA 对")
        sys.exit(1)

    # 3. 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"数据集已保存到: {args.output}")

    # 打印样例
    print("\n" + "=" * 60)
    print(f"  生成了 {len(qa_pairs)} 条评估数据")
    print("=" * 60)
    for qa in qa_pairs[:3]:
        print(f"  Q: {qa['question'][:60]}...")
        print(f"  A: {qa['ground_truth'][:60]}...")
        print()
    if len(qa_pairs) > 3:
        print(f"  ... 共 {len(qa_pairs)} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
