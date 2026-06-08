"""观测脚本: 依赖图谱场景 #2 / #3 / #4 / #6 的检索效果。

用一个内存版的 fake Milvus + fake 向量检索器, 喂入一小批构造好的 chunk
(含 related_assets 交叉引用边 / 段落顺序 / 页码), 直观对比"基础检索 vs 邻域扩展"
的命中变化, 不依赖真实 Milvus / embedding 服务。

运行:
    python -m pipeline.tests.observe_neighbor_expansion
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from pipeline.retrieval.neighbor_expansion import (
    NeighborExpander,
    apply_neighbor_expansion,
)
from pipeline.retrieval.retrievers import Hit
from pipeline.routing.decision_builder import build_from_multi_args
from pipeline.routing.fc_schema import build_plan_tool, build_multi_tool


# ---------------------------------------------------------------------------
# 内存语料 (1 篇主文献 d1: MoS2 综述; 另有 d2/d3 同类文献用于 similar)
# ---------------------------------------------------------------------------

def _c(pk, doc_id, doc_name, ctype, page, para, content, related=None):
    return {
        "pk": pk, "chunk_id": f"cid-{pk}", "doc_id": doc_id, "doc_name": doc_name,
        "type": ctype, "section": "", "page_start": page, "paragraph_index": para,
        "publication_year": 2024, "content": content, "context": "",
        "related_assets": related or [],
    }


CORPUS: List[Dict[str, Any]] = [
    # d1: 主文献, 第 2 页 (page_start=1) 有图3 + 围绕图3的段落
    _c("d1p3", "d1", "MoS2综述", "text", 1, 3, "MoS2 的能带结构随层数变化。"),
    _c("d1p4", "d1", "MoS2综述", "text", 1, 4,
       "如图3所示, 单层 MoS2 的带隙约为 1.8 eV, 属于直接带隙半导体。",
       related=[{"type": "image", "label": "Fig. 3", "chunk_id": "cid-d1fig3"}]),
    _c("d1p5", "d1", "MoS2综述", "text", 1, 5, "此外我们还研究了 MoS2 的载流子迁移率与缺陷态。"),
    _c("d1fig3", "d1", "MoS2综述", "image", 1, -1, "[Caption] Fig.3 MoS2 能带结构图",
       related=[{"type": "text", "label": "para 4", "chunk_id": "cid-d1p4"}]),
    _c("d1p9", "d1", "MoS2综述", "text", 4, 9, "结论部分与图3无关的其它讨论。"),
    # d2/d3: 同类文献 (similar 命中)
    _c("d2p1", "d2", "WS2研究", "text", 0, 1, "WS2 单层带隙约 2.0 eV, 与 MoS2 同属过渡金属硫化物。"),
    _c("d3p1", "d3", "WSe2研究", "text", 0, 1, "WSe2 的能带结构与 MoS2 类似, 可用于光电器件。"),
]

BY_CHUNK_ID = {r["chunk_id"]: r for r in CORPUS}


# ---------------------------------------------------------------------------
# Fake Milvus: 支持本模块生成的 filter 子集
# ---------------------------------------------------------------------------

class FakeMilvus:
    def query(self, collection_name, filter, output_fields=None, limit=200):
        rows = [r for r in CORPUS if _match(filter, r)]
        return rows[:limit]


def _strip_parens(atom: str) -> str:
    return atom.replace("(", "").replace(")", "").strip()


def _match(filter_expr: str, row: Dict[str, Any]) -> bool:
    if not filter_expr or not filter_expr.strip():
        return True
    atoms = [a for a in re.split(r"\s+and\s+", filter_expr) if a.strip()]
    return all(_match_atom(_strip_parens(a), row) for a in atoms)


def _match_atom(atom: str, row: Dict[str, Any]) -> bool:
    m = re.match(r'^(\w+)\s+in\s+\[(.*)\]$', atom)
    if m:
        field, body = m.group(1), m.group(2)
        items = [x.strip() for x in body.split(",") if x.strip()]
        for it in items:
            if it.startswith('"') and it.endswith('"'):
                if str(row.get(field)) == it[1:-1]:
                    return True
            else:
                try:
                    if int(row.get(field)) == int(it):
                        return True
                except (TypeError, ValueError):
                    pass
        return False
    m = re.match(r'^(\w+)\s*==\s*"(.*)"$', atom)
    if m:
        return str(row.get(m.group(1))) == m.group(2)
    m = re.match(r'^(\w+)\s*==\s*(-?\d+)$', atom)
    if m:
        try:
            return int(row.get(m.group(1))) == int(m.group(2))
        except (TypeError, ValueError):
            return False
    m = re.match(r'^(\w+)\s*>=\s*(-?\d+)$', atom)
    if m:
        try:
            return int(row.get(m.group(1)) or 0) >= int(m.group(2))
        except (TypeError, ValueError):
            return False
    m = re.match(r'^(\w+)\s*<=\s*(-?\d+)$', atom)
    if m:
        try:
            return int(row.get(m.group(1)) or 0) <= int(m.group(2))
        except (TypeError, ValueError):
            return False
    # 不认识的 atom (如 content like ...) → 不限制
    return True


class FakeVec:
    """fake more-like-this: 返回与 anchor 跨文献的同类 chunk。"""

    def retrieve(self, text, top_k=5, filter_expr=None):
        out = []
        for r in CORPUS:
            if r["doc_id"] != "d1":  # 跨文献的同类
                out.append(_row_to_hit(r))
        return out[:top_k]


def _row_to_hit(r: Dict[str, Any]) -> Hit:
    return Hit(
        pk=r["pk"], chunk_id=r["chunk_id"], doc_id=r["doc_id"], doc_name=r["doc_name"],
        type=r["type"], page_start=r["page_start"], paragraph_index=r["paragraph_index"],
        content=r["content"], related_assets=r["related_assets"],
    )


# ---------------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------------

def _fmt(hit: Hit) -> str:
    src = ",".join(hit.sources) if hit.sources else "-"
    loc = f"p{hit.page_start + 1}/para{hit.paragraph_index}"
    return f"    [{hit.type:<5} {loc:<12} src={src:<18}] {hit.content[:42]}"


def _print_block(title: str, seeds: List[Hit], neighbors: List[Hit]):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"  基础检索种子 ({len(seeds)}):")
    for h in seeds:
        print(_fmt(h))
    print(f"  ── 邻域扩展回填 ({len(neighbors)}):")
    if not neighbors:
        print("    (无)")
    for h in neighbors:
        print(_fmt(h))


def main():
    client = FakeMilvus()
    exp = NeighborExpander(client, "lit", vector_retriever=FakeVec(),
                           adjacent_window=1, page_window=0)

    # ── #4 跨模态: "图3附近的文字" → metadata(figs=[3]) + expand=[page, assets] ──
    fig_seed = _row_to_hit(BY_CHUNK_ID["cid-d1fig3"])
    fig_seed.sources = ["metadata"]
    n4 = exp.expand([fig_seed], ["page", "assets"])
    _print_block(
        "#4 跨模态: '图3附近的文字'  (种子=图3 image chunk; expand=[page,assets])",
        [fig_seed], n4,
    )

    # ── #3 邻域探索: "这篇还研究了什么" → local + expand=[adjacent, assets] ──
    para_seed = _row_to_hit(BY_CHUNK_ID["cid-d1p4"])
    para_seed.sources = ["local"]
    n3 = exp.expand([para_seed], ["adjacent", "assets"])
    _print_block(
        "#3 邻域探索: '这篇还研究了什么/相关内容'  (种子=para4; expand=[adjacent,assets])",
        [para_seed], n3,
    )

    # ── #6 同义扩展: "其他类似方法" → progressive + expand=[similar] ──
    sim_seed = _row_to_hit(BY_CHUNK_ID["cid-d1p4"])
    sim_seed.sources = ["progressive"]
    n6 = exp.expand([sim_seed], ["similar"])
    _print_block(
        "#6 同义扩展: '其他类似的研究/方法'  (种子=MoS2带隙段; expand=[similar])",
        [sim_seed], n6,
    )

    # ── 端到端: apply_neighbor_expansion 把邻居挂进 route_results ──
    from pipeline.retrieval.agentic import LocalRetrieveResult
    route_results = {"local": LocalRetrieveResult(chunk_hits=[para_seed])}
    merged = apply_neighbor_expansion(route_results, modes=["adjacent", "assets"], expander=exp)
    print(f"\n{'=' * 78}\napply_neighbor_expansion 后的 route_results 键: {list(merged.keys())}")
    print(f"  neighbor 路由命中数: {len(merged.get('neighbor', []))}")

    # ── #2 多跳/比较: "MoS2和WS2谁带隙更大" → multi 拆 2 sub ──
    print(f"\n{'=' * 78}\n#2 多跳/比较: 'MoS2 和 WS2 谁的带隙更大'  (multi 拆 2 sub)\n{'=' * 78}")
    multi_args = {
        "subs": [
            {"paths": [{"t": "progressive", "kw": ["MoS2", "带隙", "band gap"]}], "id": "sub1"},
            {"paths": [{"t": "progressive", "kw": ["WS2", "带隙", "band gap"]}], "id": "sub2"},
        ],
        "synth": "对比两者带隙数值",
    }
    multi = build_from_multi_args(multi_args, query="MoS2和WS2谁的带隙更大")
    for sub in multi.subqueries:
        print(f"  {sub.id}: routes={sub.decision.routes} rewrites={sub.decision.rewrites}")
    print(f"  synth_hint: {multi.synth_hint!r}")

    # ── 确认 FC schema 已暴露 expand 字段 ──
    plan_props = build_plan_tool()["function"]["parameters"]["properties"]
    path_props = plan_props["paths"]["items"]["oneOf"]
    has_expand = [
        p.get("properties", {}).get("t", {}).get("const")
        for p in path_props if "expand" in p.get("properties", {})
    ]
    print(f"\n{'=' * 78}\nFC schema: 暴露 expand 字段的路径类型 = {has_expand}")


if __name__ == "__main__":
    main()
