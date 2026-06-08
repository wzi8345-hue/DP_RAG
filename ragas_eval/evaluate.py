"""RAGAS 评估脚本: 评估 DP-RAG Pipeline 的检索与生成质量。

用法:
    python evaluate.py                          # 默认使用 config.yaml
    python evaluate.py --config my_config.yaml  # 指定配置
    python evaluate.py --mode full              # 覆盖模式为全链路评估

输出:
    results/ 目录下生成:
    - metrics_report.json   详细指标结果
    - metrics_report.csv    可读的表格结果
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_eval_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str, base_dir: Path) -> str:
    """将评测配置中的相对路径解析为相对 config.yaml 所在目录的绝对路径。"""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def resolve_eval_config_path(config_arg: str) -> Path:
    """兼容从项目根目录或 ragas_eval 目录运行 evaluate.py。"""
    p = Path(config_arg).expanduser()
    if p.is_absolute():
        return p
    cwd_path = p.resolve()
    if cwd_path.exists():
        return cwd_path
    script_dir_path = (Path(__file__).resolve().parent / p).resolve()
    if script_dir_path.exists():
        return script_dir_path
    return cwd_path


def build_pipeline_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构建传给 pipeline.load_config 的运行时覆盖项。"""
    milvus_override = cfg.get("milvus_override") or cfg.get("milvus") or {}
    if not isinstance(milvus_override, dict) or not milvus_override:
        return {}
    override = {"milvus": dict(milvus_override)}
    backend = str(override["milvus"].get("backend") or "server").strip().lower()
    override["milvus"]["backend"] = backend
    if backend == "server":
        server = override["milvus"].setdefault("server", {})
        if not isinstance(server, dict):
            server = {}
            override["milvus"]["server"] = server
        uri = server.get("uri") or override["milvus"].get("uri")
        if not uri or str(uri).endswith(".db") or "://" not in str(uri):
            raise ValueError(
                "ragas_eval 需要远程 Milvus: 请在 config.yaml 的 "
                "milvus_override.server.uri 配置 http(s):// 或 grpc:// 地址"
            )
    return override


def contexts_from_route_results(route_results: Dict[str, Any], fallback_context: str = "") -> List[str]:
    """从 agentic/langgraph 的纯检索结果中提取 chunk 文本。"""
    contexts: List[str] = []
    seen = set()

    def _add(text: Any) -> None:
        if not isinstance(text, str):
            return
        text = text.strip()
        if not text or text in seen:
            return
        seen.add(text)
        contexts.append(text)

    for res in (route_results or {}).values():
        hits = getattr(res, "chunk_hits", None)
        if hits is None and isinstance(res, list):
            hits = res
        for hit in hits or []:
            _add(getattr(hit, "content", ""))
            if getattr(hit, "type", "") in ("image", "table"):
                _add(getattr(hit, "context", ""))

    if not contexts and fallback_context:
        _add(fallback_context)
    return contexts


# ---------------------------------------------------------------------------
# Pipeline 调用: 获取检索结果和生成答案
# ---------------------------------------------------------------------------

# eval_kind 决定评测方式:
#   fact_retrieval       → 全库 agentic 检索 (问题自包含), 看能否召回 gold 块
#   doc_scoped_retrieval → 硬过滤到 doc_id 后库内检索 (隔离"先定位哪篇"的干扰)
#   generation / skip_auto → 不做检索召回打分
RETRIEVAL_KINDS = {"fact_retrieval", "doc_scoped_retrieval"}

_QTYPE_TO_EVAL_KIND = {
    "progressive": "fact_retrieval",
    "local": "doc_scoped_retrieval",
    "metadata_fig": "doc_scoped_retrieval",
    "metadata_page": "doc_scoped_retrieval",
    "metadata_entity": "doc_scoped_retrieval",
    "references": "doc_scoped_retrieval",
    "summary": "generation",
    "multi": "skip_auto",
    "ambiguous": "skip_auto",
}


def eval_kind_of(item: Dict[str, Any]) -> str:
    """优先用数据集里的 eval_kind; 缺失则按 query_type 兜底推断。"""
    raw = str(item.get("eval_kind") or "").strip().lower()
    if raw in {"fact_retrieval", "doc_scoped_retrieval", "generation", "skip_auto"}:
        return raw
    return _QTYPE_TO_EVAL_KIND.get(item.get("query_type", "progressive"), "fact_retrieval")


def run_pipeline(
    items: List[Dict[str, Any]],
    pipeline_config_path: str,
    pipeline_overrides: Optional[Dict[str, Any]] = None,
    mode: str = "retrieval",
    top_k: int = 8,
    retrieval_only: bool = True,
) -> List[Dict[str, Any]]:
    """按 eval_kind 分流调用检索, 收集召回上下文 (+ full 模式收答案)。"""
    # 延迟导入, 避免在仅编辑数据集时加载 heavy 依赖
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import load_config
    from pipeline.flows.query import QueryFlow

    config = load_config(pipeline_config_path, overrides=pipeline_overrides)
    flow = QueryFlow(config)

    # doc_scoped 用简单检索器 + 硬 doc_id 过滤 (懒加载一次)
    _cache: Dict[str, Any] = {}

    def _hard_filter_retrieve(q: str, doc_id: str, query_type: str = "") -> List[str]:
        if "hybrid" not in _cache:
            _meta, _vec, _bm25, hybrid = flow._get_simple_retrievers()
            _cache["hybrid"] = hybrid
            _cache["meta"] = _meta
        hybrid = _cache["hybrid"]
        filter_expr = f'doc_id == "{doc_id}"'

        # metadata_fig (问'图N/表N 展示了什么'): 评测此前用纯语义 hybrid, 但生产对
        # '图N/表N' 实际走 metadata 路由 (MetadataRetriever: 解析图号 → content LIKE
        # '%图 3%' + score_fig_table_refs 打分, image 命中 +5)。评测不复刻这条路径就会
        # 系统性低估 metadata_fig (语义检索被该篇标题/摘要顶满)。
        # 这里**直接调用生产 MetadataRetriever** (不修改任何生产检索代码), 让评测忠实反映
        # 生产能力; 图号无命中时 (gold 不含标签/解析失败) 回退到结构块语义检索, 保证不劣于现状。
        if query_type == "metadata_fig":
            meta = _cache["meta"]
            try:
                ref_hits = meta.retrieve(
                    q, top_k=top_k,
                    filter_expr=filter_expr,
                    max_candidates=max(50, top_k * 5),
                )
            except Exception as e:
                logger.warning(f"  metadata_fig 复刻生产路由失败, 回退语义: {e}")
                ref_hits = []
            ref_ctx = [
                getattr(h, "content", "") for h in ref_hits if getattr(h, "content", "")
            ]
            if ref_ctx:
                return ref_ctx[:top_k]
            # 图号未命中: 回退到该篇结构块 (image/table) 语义检索
            filter_expr += ' and (type == "image" or type == "table")'

        hits = hybrid.retrieve(
            q, top_k=top_k,
            filter_expr=filter_expr,
            per_retriever_k=max(20, top_k * 3),
        )
        return [getattr(h, "content", "") for h in hits if getattr(h, "content", "")]

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        q = item["question"]
        kind = eval_kind_of(item)
        doc_id = item.get("doc_id")
        logger.info(f"[{i+1}/{len(items)}] ({kind}) {q[:60]}")
        t0 = time.time()
        try:
            if kind == "doc_scoped_retrieval" and doc_id:
                contexts = _hard_filter_retrieve(
                    q, str(doc_id), item.get("query_type", ""),
                )
                answer, err, rmode = "", None, "hard_filter"
            else:
                # fact_retrieval: 全库 agentic 单轮 (问题须自包含)。评测检索效率时
                # 只跑 agent.run()/pipeline.run() 的检索与 context_build, 不调用最终总结生成 LLM。
                if retrieval_only:
                    lg_cfg = config.retrieval.get("langgraph", {}) or {}
                    if bool(lg_cfg.get("enabled", False)):
                        retrieval_result = flow._get_langgraph_agent().run(q, session_meta={})
                        rmode = "langgraph_retrieval_only"
                    else:
                        retrieval_result = flow._get_agentic_pipeline().run(q)
                        rmode = "agentic_retrieval_only"
                    contexts = contexts_from_route_results(
                        retrieval_result.get("results", {}),
                        retrieval_result.get("context", ""),
                    )
                    latency = float((retrieval_result.get("latency") or {}).get("total_s") or 0)
                    answer, err = "", None
                else:
                    qr, _session = flow.run(q, use_agentic=True)
                    contexts = [
                        h.get("content", "")
                        for h in qr.hits
                        if isinstance(h, dict) and h.get("content")
                    ]
                    if not contexts and qr.context:
                        contexts = [qr.context]
                    latency = round(time.time() - t0, 3)
                    answer, err, rmode = (qr.answer or ""), qr.error, "agentic"

            results.append({
                "question": q,
                "contexts": contexts,
                "answer": answer if mode == "full" else "",
                "latency_s": latency if retrieval_only and kind != "doc_scoped_retrieval" else round(time.time() - t0, 3),
                "error": err,
                "retrieval_mode": rmode,
            })
            logger.info(f"  → {len(contexts)} 条上下文, {round(time.time()-t0,2)}s")
        except Exception as e:
            logger.error(f"  ✗ Pipeline 调用失败: {e}")
            results.append({
                "question": q,
                "contexts": [],
                "answer": "",
                "latency_s": 0,
                "error": str(e),
                "retrieval_mode": "error",
            })

    return results


# ---------------------------------------------------------------------------
# Gold-context 命中率 (确定性, 不依赖 LLM/API)
#
# 因检索库与 gold (mineru) 的切块/chunk_id 不一致, 无法按 id 精确匹配, 改用文本重叠:
# gold 文本的任一长度 >= window 的连续片段出现在任一召回 chunk 里, 即判为命中。
# ---------------------------------------------------------------------------

def _norm_text(t: str) -> str:
    """归一化用于文本重叠比较: 全角→半角(NFKC)、小写、统一乘号、删所有空白。

    科研文献里同一数值在 gold 与召回 chunk 中常因排版不同 (全角数字、× 与 x、
    "0.047 8" 这种数字内空格) 而无法精确匹配, 不归一化会把"其实召回到了"误判成 0。
    """
    s = unicodedata.normalize("NFKC", t or "").lower()
    s = s.replace("×", "x").replace("∼", "~").replace("－", "-")
    return re.sub(r"\s+", "", s)


def gold_context_recall(
    gold_texts: List[str],
    retrieved_texts: List[str],
    window: int = 20,
) -> Optional[float]:
    """gold 块被召回的比例; 无 gold 返回 None (该条不计入均值)。"""
    golds = [g for g in (gold_texts or []) if isinstance(g, str) and _norm_text(g)]
    if not golds:
        return None
    haystack = _norm_text(" ".join(retrieved_texts or []))
    if not haystack:
        return 0.0
    hit = 0
    for g in golds:
        ng = _norm_text(g)
        if len(ng) <= window:
            ok = ng in haystack
        else:
            step = max(1, window // 2)
            ok = any(ng[i:i + window] in haystack for i in range(0, len(ng) - window + 1, step))
        hit += 1 if ok else 0
    return hit / len(golds)


def exact_answer_hit(ground_truth: str, retrieved_texts: List[str]) -> Optional[float]:
    """ground_truth 是否能在召回文本里直接命中 (补 gold-context 测不了的短答案/数字/名称)。

    短答案 (<=30字): 整串子串匹配; 长答案: 取前 20 字连续片段命中即可。无 GT 返回 None。
    """
    gt = _norm_text(ground_truth)
    if not gt:
        return None
    haystack = _norm_text(" ".join(retrieved_texts or []))
    if not haystack:
        return 0.0
    if len(gt) <= 30:
        return 1.0 if gt in haystack else 0.0
    return 1.0 if gt[:20] in haystack else 0.0


# ---------------------------------------------------------------------------
# RAGAS 评估
# ---------------------------------------------------------------------------

def evaluate_with_ragas(
    eval_data: List[Dict[str, Any]],
    ragas_llm_cfg: Dict[str, Any],
    ragas_emb_cfg: Optional[Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Any]:
    """使用 RAGAS 计算评估指标。"""
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset

    # 构建 RAGAS LLM 和 Embeddings
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(
        api_key=ragas_llm_cfg.get("api_key", ""),
        base_url=ragas_llm_cfg.get("api_base"),
        model=ragas_llm_cfg.get("model", "gpt-4"),
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(llm)

    # 选择指标
    metric_map = {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    selected_metrics = []
    for m in metrics:
        if m in metric_map:
            metric_obj = metric_map[m]
            metric_obj.llm = ragas_llm
            # answer_relevancy 需要 embeddings
            if m == "answer_relevancy" and ragas_emb_cfg:
                embeddings = OpenAIEmbeddings(
                    api_key=ragas_emb_cfg.get("api_key", ""),
                    base_url=ragas_emb_cfg.get("api_base"),
                    model=ragas_emb_cfg.get("model", "text-embedding-ada-002"),
                )
                from ragas.embeddings import LangchainEmbeddingsWrapper
                metric_obj.embeddings = LangchainEmbeddingsWrapper(embeddings)
            selected_metrics.append(metric_obj)
        else:
            logger.warning(f"未知指标: {m}, 跳过")

    if not selected_metrics:
        raise ValueError("没有可用的评估指标")

    # 构建 HuggingFace Dataset
    data_dict = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
    }
    for item in eval_data:
        data_dict["question"].append(item["question"])
        data_dict["contexts"].append(item.get("retrieved_contexts", []))
        data_dict["answer"].append(item.get("answer", ""))
        data_dict["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(data_dict)

    # 运行评估
    logger.info(f"开始 RAGAS 评估, 指标: {[m.name for m in selected_metrics]}")
    result = evaluate(
        dataset,
        metrics=selected_metrics,
    )

    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估 DP-RAG Pipeline")
    parser.add_argument("--config", default="config.yaml", help="评估配置文件路径")
    parser.add_argument("--mode", choices=["retrieval", "full"], help="覆盖评估模式")
    parser.add_argument("--dataset", help="覆盖数据集路径")
    parser.add_argument("--output", help="覆盖输出目录")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = resolve_eval_config_path(args.config)
    config_dir = config_path.parent

    cfg = load_eval_config(str(config_path))
    mode = args.mode or cfg.get("mode", "retrieval")
    dataset_path = resolve_path(args.dataset or cfg.get("dataset", "datasets/test_dataset.json"), config_dir)
    output_dir = resolve_path(args.output or cfg.get("output_dir", "results"), config_dir)

    # 加载测试数据集
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"加载 {len(dataset)} 条测试数据 (mode={mode})")

    # ── 数据集过滤 ──────────────────────────────────────────────────────
    flt = cfg.get("filter", {}) or {}
    query_types = flt.get("query_types")          # 例 ["progressive"]; None=全部
    # 默认只评可做检索召回打分的两类 (generation/skip_auto 不进检索指标)
    eval_kinds = flt.get("eval_kinds") or sorted(RETRIEVAL_KINDS)
    in_corpus_only = bool(flt.get("in_corpus_only", True))
    gold_window = int(cfg.get("gold_overlap_window", 20))
    top_k = int(cfg.get("top_k", 8))
    retrieval_only = bool(cfg.get("retrieval_only", True))

    items = list(dataset)
    if in_corpus_only:
        before = len(items)
        items = [d for d in items if d.get("in_corpus")]
        logger.info(f"in_corpus_only=True: {before} -> {len(items)} 条")
    kset = set(eval_kinds)
    before = len(items)
    items = [d for d in items if eval_kind_of(d) in kset]
    logger.info(f"eval_kinds={sorted(kset)}: {before} -> {len(items)} 条")
    if query_types:
        before = len(items)
        qset = set(query_types)
        items = [d for d in items if d.get("query_type", "progressive") in qset]
        logger.info(f"query_types={query_types}: {before} -> {len(items)} 条")

    if not items:
        logger.error(
            "过滤后无可评测数据。请确认: (1) 数据集含 in_corpus/eval_kind 字段 "
            "(新数据集出生即带; 旧数据集需先跑 backfill_dataset.py); "
            "(2) filter.eval_kinds / query_types / in_corpus_only 设置合理。"
        )
        sys.exit(1)

    # 调用 Pipeline (按 eval_kind 分流: fact=全库agentic, doc_scoped=硬doc_id过滤)
    pipeline_cfg = resolve_path(cfg.get("pipeline_config", "../pipeline/default_config.yaml"), config_dir)
    pipeline_overrides = build_pipeline_overrides(cfg)
    pipeline_results = run_pipeline(
        items,
        pipeline_cfg,
        pipeline_overrides=pipeline_overrides,
        mode=mode,
        top_k=top_k,
        retrieval_only=retrieval_only,
    )

    # 合并: ground_truth + 召回结果 + gold-context 召回 + 精确答案命中
    eval_data: List[Dict[str, Any]] = []
    for item, result in zip(items, pipeline_results):
        recall = gold_context_recall(
            item.get("ground_contexts", []), result["contexts"], window=gold_window,
        )
        exact = exact_answer_hit(item.get("ground_truth", ""), result["contexts"])
        eval_data.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "query_type": item.get("query_type", "progressive"),
            "eval_kind": eval_kind_of(item),
            "doc_id": item.get("doc_id"),
            "retrieval_mode": result.get("retrieval_mode", ""),
            "retrieved_contexts": result["contexts"],
            "gold_context_recall": recall,
            "exact_answer_hit": exact,
            "answer": result.get("answer", "") if mode == "full" else "",
            "latency_s": result.get("latency_s", 0),
            "error": result.get("error"),
        })

    # 检查是否有成功的检索
    success_count = sum(1 for d in eval_data if not d.get("error"))
    if success_count == 0:
        logger.error("所有查询均失败, 无法评估")
        sys.exit(1)

    # RAGAS 评估
    ragas_llm_cfg = cfg.get("ragas_llm", {})
    ragas_emb_cfg = cfg.get("ragas_embeddings")
    metrics = cfg.get("metrics", ["context_precision", "context_recall"])

    # full 模式额外启用指标
    if mode == "full" and not retrieval_only:
        if "faithfulness" not in metrics:
            metrics.append("faithfulness")
        if "answer_relevancy" not in metrics:
            metrics.append("answer_relevancy")

    # RAGAS 是可选的 (需要外部 LLM/API); 失败不致命, gold-context 命中率始终可用。
    metrics_dict: Dict[str, Any] = {}
    if cfg.get("run_ragas", True):
        try:
            result = evaluate_with_ragas(eval_data, ragas_llm_cfg, ragas_emb_cfg, metrics)
            if hasattr(result, "scores") and hasattr(result.scores, "items"):
                metrics_dict = dict(result.scores)
            elif hasattr(result, "to_pandas"):
                df = result.to_pandas()
                for col in df.columns:
                    if col not in ("question",):
                        try:
                            metrics_dict[col] = float(df[col].mean())
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"RAGAS 评估失败 (跳过, 仍输出 gold-context 命中率): {e}")
    else:
        logger.info("run_ragas=False: 跳过 RAGAS, 仅用 gold-context 命中率评检索")

    # ── 聚合 (gold-context 召回为主指标; exact 为短答案补充) ──────────────
    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [r for r in rows if not r.get("error")]
        recalls = [r["gold_context_recall"] for r in rows if r.get("gold_context_recall") is not None]
        exacts = [r["exact_answer_hit"] for r in rows if r.get("exact_answer_hit") is not None]
        lats = [r.get("latency_s", 0) for r in ok]
        return {
            "count": len(rows),
            "success": len(ok),
            "gold_context_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "exact_answer_hit": round(sum(exacts) / len(exacts), 4) if exacts else None,
            "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else None,
        }

    def _group(key: str) -> Dict[str, Any]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for d in eval_data:
            buckets[d.get(key)].append(d)
        return {k: _agg(rows) for k, rows in sorted(buckets.items(), key=lambda x: str(x[0]))}

    per_eval_kind = _group("eval_kind")
    per_query_type = _group("query_type")
    overall = _agg(eval_data)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "mode": mode,
        "retrieval_only": retrieval_only,
        "dataset": dataset_path,
        "pipeline_config": pipeline_cfg,
        "pipeline_overrides": pipeline_overrides,
        "filter": {
            "query_types": query_types, "eval_kinds": sorted(kset),
            "in_corpus_only": in_corpus_only,
        },
        "num_questions": len(eval_data),
        "success_count": success_count,
        "overall": overall,
        "ragas_metrics": metrics_dict,
        "per_eval_kind": per_eval_kind,
        "per_query_type": per_query_type,
        "per_question": eval_data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "metrics_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # CSV 摘要 (按 eval_kind)
    try:
        import pandas as pd
        rows = [{"eval_kind": k, **agg} for k, agg in per_eval_kind.items()]
        rows.append({"eval_kind": "ALL", **overall})
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "metrics_report.csv"), index=False,
        )
    except ImportError:
        pass

    # 打印摘要
    def _print_table(title: str, table: Dict[str, Any]) -> None:
        print(f"\n  [{title}]")
        print(f"  {'':<22}{'n':>5}{'ok':>5}{'gold_recall':>13}{'exact':>9}{'lat(s)':>9}")
        print("  " + "-" * 63)
        for k, a in table.items():
            gr = "-" if a["gold_context_recall"] is None else f"{a['gold_context_recall']:.3f}"
            ex = "-" if a["exact_answer_hit"] is None else f"{a['exact_answer_hit']:.3f}"
            lat = "-" if a["avg_latency_s"] is None else f"{a['avg_latency_s']:.2f}"
            print(f"  {str(k):<22}{a['count']:>5}{a['success']:>5}{gr:>13}{ex:>9}{lat:>9}")

    print("\n" + "=" * 72)
    print(f"  检索评估结果 (mode={mode})  数据集={dataset_path}")
    print(f"  retrieval_only={retrieval_only} (True 时不调用最终总结生成 LLM)")
    print(f"  过滤: eval_kinds={sorted(kset)} query_types={query_types or '全部'} "
          f"in_corpus_only={in_corpus_only}")
    print("=" * 72)
    _print_table("按 eval_kind", per_eval_kind)
    _print_table("按 query_type", per_query_type)
    gr = "-" if overall["gold_context_recall"] is None else f"{overall['gold_context_recall']:.3f}"
    print(f"\n  ALL: n={overall['count']} ok={overall['success']} gold_recall={gr}")
    if metrics_dict:
        print("  RAGAS:", {k: round(v, 4) for k, v in metrics_dict.items() if isinstance(v, (int, float))})
    print(f"  结果已保存到: {output_dir}/")
    print("=" * 72)


def _save_raw_results(eval_data: List[Dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "raw_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    logger.info(f"原始结果已保存到: {path}")


if __name__ == "__main__":
    main()
