"""真实 Milvus 观测 Progressive 双路 top-K 路由。

跑法: cd .. && PYTHONPATH=. python -m pipeline.tests.observe_progressive_live
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)

QUERY = "离子镀 炮管 防烧蚀 可行性 离子镀工艺 炮管涂层 热防护"


def main() -> None:
    from pipeline.config import load_config
    from pipeline.flows.query import QueryFlow

    flow = QueryFlow(load_config())
    cfg = flow.config.get("retrieval.progressive") or {}
    local_r = flow._get_agentic_pipeline().local_r

    print("=" * 72)
    print(f"Query: {QUERY}")
    print(
        f"Config: l1_per_path={cfg.get('level1_per_retriever_k', 5)} "
        f"l2_per_path={cfg.get('l2_per_path_k', 10)} "
        f"l2_min_score={cfg.get('l2_drill_min_score', 0.55)}"
    )
    print("=" * 72)

    result = local_r.retrieve(QUERY, top_k_docs=5, top_k_chunks=8, per_retriever_k=10)

    print(f"\n候选 doc ({len(result.candidate_docs)}):")
    for i, d in enumerate(result.candidate_docs):
        print(f"  #{i + 1} score={d.rrf_score:.4f} {d.doc_name}")

    print(f"\nL2 chunk hits ({len(result.chunk_hits)}):")
    for i, h in enumerate(result.chunk_hits[:12]):
        snip = (h.content or "")[:70].replace("\n", " ")
        print(
            f"  #{i} emb={h.score:.4f} doc={h.doc_name} sources={h.sources}\n"
            f"       {snip}..."
        )

    ion = [h for h in result.chunk_hits if "离子镀" in (h.content or "") or "炮管" in (h.content or "")]
    print(f"\n含离子镀/炮管: {len(ion)}/{len(result.chunk_hits)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
