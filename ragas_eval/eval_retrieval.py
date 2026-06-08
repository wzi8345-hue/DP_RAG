"""检索专项评测 (只检索, 不生成) —— 把"检索质量"从"agent 编排 + 生成延迟"里剥离出来。

与 evaluate.py 的区别:
  - evaluate.py: fact 走完整 agentic (router LLM + reflect + 生成), 端到端、慢 (~15s/条),
    测的是"agent 产品"的表现。
  - 本脚本: 直接调底层 HybridRetriever.retrieve (无 router LLM、无 reflect、无生成),
    只发 1 次 embedding, 亚秒级/条, 测的是"检索器召回上限"(rerank 之前的候选质量)。
    recall@k 上不去, 说明候选阶段就丢了 gold, 后面 rerank/生成再强也救不回来。

分流 (沿用 eval_kind):
  - fact_retrieval       → 全库召回 (filter=None)
  - doc_scoped_retrieval → 硬过滤 doc_id 后库内召回

指标:
  - recall@k (k=5/10/20): gold 块被召回的比例 (文本重叠, 见 evaluate.gold_context_recall)
  - exact@maxk: ground_truth 是否直接出现在召回里 (补数值/名称类)
  - table_gold 命中诊断: 区分 gold 是"表格块"还是"散文块", 量化表格检索短板

用法:
    python3 eval_retrieval.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 复用 evaluate.py 里已经写好的纯函数 (导入不会触发它的 main)
from evaluate import (
    RETRIEVAL_KINDS,
    eval_kind_of,
    exact_answer_hit,
    gold_context_recall,
)

logger = logging.getLogger(__name__)

DEFAULT_KS = [5, 10, 20]


def _is_table_gold(gold_texts: List[str]) -> bool:
    """gold 块是否为表格类 (含 [Table HTML] / <table / [Table ...])。"""
    blob = " ".join(gold_texts or [])
    return ("[Table HTML]" in blob) or ("<table" in blob.lower()) or ("[Table" in blob)


def _build_retriever(pipeline_config_path: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import load_config
    from pipeline.flows.query import QueryFlow

    flow = QueryFlow(load_config(pipeline_config_path))
    _meta, _vec, _bm25, hybrid = flow._get_simple_retrievers()
    return hybrid, flow.config


def _build_reranker(config):
    """从 pipeline 配置 (retrieval.langgraph.reranker) 构建 RerankerClient。

    无论 enabled 是否为 True 都强制构建 (专项评测显式开启 rerank), fail_open=True
    保证 rerank 服务不可用时静默回退到原始检索序, 不污染结果。
    """
    from pipeline.clients.reranker import RerankerClient

    rr = (config.retrieval.get("langgraph", {}) or {}).get("reranker", {}) or {}
    return RerankerClient(
        api_base=rr.get("api_base", "http://localhost:8001/v1"),
        model=rr.get("model", "Qwen/Qwen3-Reranker-0.6B"),
        top_k=int(rr.get("top_k", 5)),
        timeout=int(rr.get("timeout", 60)),
        max_retries=int(rr.get("max_retries", 2)),
        fail_open=True,
    )


def retrieve_only(
    items: List[Dict[str, Any]],
    pipeline_config_path: str,
    max_k: int,
    rerank: bool = False,
    rerank_candidates: int = 40,
) -> List[Dict[str, Any]]:
    """对每条 QA 只做一次底层检索, 返回 top-max_k 的召回文本 (有序)。

    用 infer_hybrid_weights 按查询算 dense/bm25 权重 (含取值型 bias), 让检索专项
    评测反映真实加权, 而非固定 0.7/0.3。

    rerank=True 时: 先召回更大候选池 (rerank_candidates), 过 reranker 重排, 再取
    top-max_k。这能验证"召回到了 top-N 但排不进 top-k"的差距是否由排序导致 (production
    链路本就带 rerank), 而非召回能力本身的问题。
    """
    hybrid, config = _build_retriever(pipeline_config_path)  # 内部 sys.path.insert, 必须先调
    from pipeline.retrieval.hybrid_weights import STAGE_SIMPLE, infer_hybrid_weights
    reranker = _build_reranker(config) if rerank else None
    # rerank 时多召回候选给重排腾挪空间, 让 recall@max_k 也能受益
    fetch_k = max(rerank_candidates, max_k) if rerank else max_k
    per_k = max(40, fetch_k * 3)

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        q = item["question"]
        kind = eval_kind_of(item)
        doc_id = item.get("doc_id")
        filter_expr = (
            f'doc_id == "{doc_id}"'
            if (kind == "doc_scoped_retrieval" and doc_id)
            else None
        )
        w = infer_hybrid_weights(STAGE_SIMPLE, q)
        t0 = time.time()
        try:
            hits = hybrid.retrieve(
                q, top_k=fetch_k, filter_expr=filter_expr, per_retriever_k=per_k,
                dense_weight=w.dense, bm25_weight=w.bm25,
            )
            contexts = [getattr(h, "content", "") for h in hits if getattr(h, "content", "")]
            if reranker is not None and contexts:
                ranked = reranker.rerank(q, contexts, top_k=max_k)
                if ranked:  # fail_open 返回 [] 时保留原始检索序
                    contexts = [r.content for r in ranked]
            contexts = contexts[:max_k]
            err = None
        except Exception as e:
            contexts, err = [], str(e)
        out.append({
            "contexts": contexts,
            "latency_s": round(time.time() - t0, 3),
            "error": err,
        })
        if (i + 1) % 20 == 0:
            logger.info(f"  检索进度 {i+1}/{len(items)}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="检索专项评测 (只检索不生成)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    dataset_path = cfg["dataset"]
    pipeline_cfg = cfg.get("pipeline_config", "../pipeline/default_config.yaml")
    gold_window = int(cfg.get("gold_overlap_window", 20))
    ks = sorted(set(cfg.get("retrieval_ks", DEFAULT_KS)))
    max_k = max(ks)
    output_dir = cfg.get("output_dir", "results")

    rerank_eval = bool(cfg.get("rerank_eval", False))
    rerank_candidates = int(cfg.get("rerank_candidates", 40))

    flt = cfg.get("filter", {}) or {}
    eval_kinds = set(flt.get("eval_kinds") or sorted(RETRIEVAL_KINDS))
    query_types = flt.get("query_types")
    in_corpus_only = bool(flt.get("in_corpus_only", True))

    dataset = json.load(open(dataset_path, "r", encoding="utf-8"))
    items = list(dataset)
    if in_corpus_only:
        items = [d for d in items if d.get("in_corpus")]
    items = [d for d in items if eval_kind_of(d) in eval_kinds]
    if query_types:
        qset = set(query_types)
        items = [d for d in items if d.get("query_type", "progressive") in qset]
    if not items:
        logger.error("过滤后无可评测数据, 检查 dataset / filter。")
        sys.exit(1)
    logger.info(
        f"检索专项评测: {len(items)} 条 | ks={ks} | "
        f"rerank={'ON(cand=%d)' % rerank_candidates if rerank_eval else 'OFF'} | "
        f"dataset={dataset_path}"
    )

    results = retrieve_only(
        items, pipeline_cfg, max_k=max_k,
        rerank=rerank_eval, rerank_candidates=rerank_candidates,
    )

    # 逐条算 recall@k / exact / table 诊断
    per_question: List[Dict[str, Any]] = []
    for item, res in zip(items, results):
        gold = item.get("ground_contexts", [])
        ctx = res["contexts"]
        recalls = {
            f"recall@{k}": gold_context_recall(gold, ctx[:k], window=gold_window)
            for k in ks
        }
        per_question.append({
            "question": item["question"],
            "eval_kind": eval_kind_of(item),
            "query_type": item.get("query_type", "progressive"),
            "doc_id": item.get("doc_id"),
            "is_table_gold": _is_table_gold(gold),
            "n_retrieved": len(ctx),
            **recalls,
            f"exact@{max_k}": exact_answer_hit(item.get("ground_truth", ""), ctx),
            "latency_s": res["latency_s"],
            "error": res["error"],
        })

    # 聚合
    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _mean(field: str) -> Optional[float]:
            vals = [r[field] for r in rows if r.get(field) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        tbl = [r for r in rows if r["is_table_gold"]]
        prose = [r for r in rows if not r["is_table_gold"]]
        agg: Dict[str, Any] = {
            "count": len(rows),
            "n_table_gold": len(tbl),
            "n_prose_gold": len(prose),
        }
        for k in ks:
            agg[f"recall@{k}"] = _mean(f"recall@{k}")
        agg[f"exact@{max_k}"] = _mean(f"exact@{max_k}")
        # 分表格/散文的主指标 (recall@max_k), 直接量化表格检索短板
        agg[f"table_recall@{max_k}"] = (
            round(sum(r[f"recall@{max_k}"] or 0 for r in tbl) / len(tbl), 4) if tbl else None
        )
        agg[f"prose_recall@{max_k}"] = (
            round(sum(r[f"recall@{max_k}"] or 0 for r in prose) / len(prose), 4) if prose else None
        )
        agg["avg_latency_s"] = _mean("latency_s")
        return agg

    by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_question:
        by_kind[r["eval_kind"]].append(r)
    per_eval_kind = {k: _agg(v) for k, v in sorted(by_kind.items())}
    overall = _agg(per_question)

    os.makedirs(output_dir, exist_ok=True)
    report = {
        "kind": "retrieval_only",
        "dataset": dataset_path,
        "ks": ks,
        "rerank_eval": rerank_eval,
        "rerank_candidates": rerank_candidates if rerank_eval else None,
        "filter": {"eval_kinds": sorted(eval_kinds), "query_types": query_types,
                   "in_corpus_only": in_corpus_only},
        "overall": overall,
        "per_eval_kind": per_eval_kind,
        "per_question": per_question,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_json = os.path.join(output_dir, "retrieval_report.json")
    json.dump(report, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 打印
    rk = f"recall@{max_k}"
    print("\n" + "=" * 78)
    print(f"  检索专项评测 (只检索不生成)  dataset={dataset_path}")
    print(f"  rerank: {'ON  (候选池=%d → 重排取 top-%d)' % (rerank_candidates, max_k) if rerank_eval else 'OFF (rerank 之前的候选质量)'}")
    print("=" * 78)
    hdr = f"  {'eval_kind':<22}{'n':>4}" + "".join(f"{'R@'+str(k):>8}" for k in ks)
    hdr += f"{'exact':>8}{'tblR':>8}{'prosR':>8}{'lat':>7}"
    print(hdr)
    print("  " + "-" * 74)

    def _row(name: str, a: Dict[str, Any]) -> None:
        def f(v): return "-" if v is None else f"{v:.3f}"
        line = f"  {name:<22}{a['count']:>4}" + "".join(f"{f(a[f'recall@{k}']):>8}" for k in ks)
        line += f"{f(a[f'exact@{max_k}']):>8}{f(a[f'table_recall@{max_k}']):>8}"
        line += f"{f(a[f'prose_recall@{max_k}']):>8}{(a['avg_latency_s'] or 0):>7.2f}"
        print(line)
        if a["n_table_gold"]:
            print(f"      └ gold 分布: 表格 {a['n_table_gold']} / 散文 {a['n_prose_gold']}")

    for k, a in per_eval_kind.items():
        _row(k, a)
    print("  " + "-" * 74)
    _row("ALL", overall)
    print(f"\n  说明: tblR/prosR = {rk} 在'表格类 gold'/'散文类 gold'上的召回; "
          f"二者差距越大, 表格检索短板越明显。")
    print(f"  结果: {out_json}")
    print("=" * 78)


if __name__ == "__main__":
    main()
