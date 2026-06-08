"""邻域扩展 (依赖图谱场景 #3/#4/#6) 单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.retrieval.neighbor_expansion import (
    EXPAND_ADJACENT,
    EXPAND_ASSETS,
    EXPAND_PAGE,
    EXPAND_SIMILAR,
    ROUTE_NEIGHBOR,
    NeighborExpander,
    apply_neighbor_expansion,
    collect_expand_modes,
    collect_seed_hits,
    normalize_expand_modes,
)
from pipeline.retrieval.agentic import LocalRetrieveResult
from pipeline.retrieval.retrievers import Hit
from pipeline.models import RouteDecision
from pipeline.routing.decision_builder import build_from_plan_args


def _row(pk: str, *, doc_id="d1", ctype="text", content="", page=0, para=1,
         related=None) -> dict:
    return {
        "pk": pk,
        "chunk_id": f"cid-{pk}",
        "doc_id": doc_id,
        "doc_name": doc_id,
        "type": ctype,
        "section": "",
        "page_start": page,
        "paragraph_index": para,
        "publication_year": 2024,
        "content": content,
        "context": "",
        "related_assets": related or [],
    }


def _seed(pk: str, *, doc_id="d1", ctype="text", page=0, para=1, related=None) -> Hit:
    return Hit(
        pk=pk, chunk_id=f"cid-{pk}", doc_id=doc_id, doc_name=doc_id, type=ctype,
        page_start=page, paragraph_index=para, content=f"seed {pk}",
        related_assets=related or [],
    )


class TestNormalize(unittest.TestCase):
    def test_normalize_filters_invalid_and_dedupes(self):
        self.assertEqual(
            normalize_expand_modes(["assets", "ASSETS", "bogus", "page"]),
            ["assets", "page"],
        )
        self.assertEqual(normalize_expand_modes("similar"), ["similar"])
        self.assertEqual(normalize_expand_modes(None), [])

    def test_collect_expand_modes_across_decisions(self):
        d1 = RouteDecision(expand_neighbors=["adjacent"])
        d2 = RouteDecision(expand_neighbors=["assets", "adjacent"])
        self.assertEqual(collect_expand_modes([d1, d2]), ["adjacent", "assets"])


class TestSeedCollection(unittest.TestCase):
    def test_collect_from_mixed_results(self):
        res = {
            "progressive": LocalRetrieveResult(chunk_hits=[_seed("a"), _seed("b")]),
            "metadata": [_seed("c")],
            ROUTE_NEIGHBOR: [_seed("z")],  # 已有 neighbor 不算种子
        }
        seeds = collect_seed_hits(res)
        self.assertEqual({h.pk for h in seeds}, {"a", "b", "c"})


class TestExpandAssets(unittest.TestCase):
    def test_follows_related_assets_chunk_ids(self):
        seed = _seed("s1", related=[
            {"type": "image", "label": "Fig. 3", "chunk_id": "cid-img3"},
            {"type": "equation", "label": "Eq.", "chunk_id": "cid-eq1"},
        ])
        client = MagicMock()
        client.query.return_value = [
            _row("img3", ctype="image", content="Fig.3 caption"),
            _row("eq1", ctype="equation", content="$$E=mc^2$$"),
        ]
        exp = NeighborExpander(client, "lit")
        out = exp.expand([seed], [EXPAND_ASSETS])
        self.assertEqual({h.pk for h in out}, {"img3", "eq1"})
        # filter 用 chunk_id in [...]
        flt = client.query.call_args.kwargs["filter"]
        self.assertIn("chunk_id in [", flt)
        self.assertIn("cid-img3", flt)

    def test_no_assets_no_query(self):
        client = MagicMock()
        exp = NeighborExpander(client, "lit")
        out = exp.expand([_seed("s1")], [EXPAND_ASSETS])
        self.assertEqual(out, [])
        client.query.assert_not_called()


class TestExpandAdjacent(unittest.TestCase):
    def test_adjacent_paragraph_window(self):
        seed = _seed("s1", doc_id="d1", para=5)
        client = MagicMock()
        client.query.return_value = [
            _row("p4", doc_id="d1", para=4, content="prev"),
            _row("p6", doc_id="d1", para=6, content="next"),
        ]
        exp = NeighborExpander(client, "lit", adjacent_window=1)
        out = exp.expand([seed], [EXPAND_ADJACENT])
        self.assertEqual({h.pk for h in out}, {"p4", "p6"})
        flt = client.query.call_args.kwargs["filter"]
        self.assertIn('doc_id == "d1"', flt)
        self.assertIn("paragraph_index in [4, 5, 6]", flt)

    def test_seed_excluded_from_neighbors(self):
        seed = _seed("s1", doc_id="d1", para=5)
        client = MagicMock()
        # query 返回里混入种子自身 (cid-s1) → 应被去重
        client.query.return_value = [
            _row("p4", doc_id="d1", para=4),
            {"pk": "s1", "chunk_id": "cid-s1", "doc_id": "d1", "type": "text",
             "page_start": 0, "paragraph_index": 5, "publication_year": 2024,
             "content": "seed", "context": "", "related_assets": [], "doc_name": "d1",
             "section": ""},
        ]
        exp = NeighborExpander(client, "lit")
        out = exp.expand([seed], [EXPAND_ADJACENT])
        self.assertEqual({h.pk for h in out}, {"p4"})


class TestExpandPage(unittest.TestCase):
    def test_same_page_text_near_figure(self):
        # 场景 #4: 图3 (image chunk) 在第 2 页 (page_start=1), 找同页文字
        fig_seed = _seed("fig3", doc_id="d1", ctype="image", page=1, para=-1)
        client = MagicMock()
        client.query.return_value = [
            _row("t1", doc_id="d1", ctype="text", page=1, content="围绕图3的论述"),
        ]
        exp = NeighborExpander(client, "lit", page_window=0)
        out = exp.expand([fig_seed], [EXPAND_PAGE])
        self.assertEqual({h.pk for h in out}, {"t1"})
        flt = client.query.call_args.kwargs["filter"]
        self.assertIn("page_start in [1]", flt)


class TestExpandSimilar(unittest.TestCase):
    def test_uses_vector_retriever(self):
        seed = _seed("s1", doc_id="d1")
        vec = MagicMock()
        vec.retrieve.return_value = [_seed("sim1", doc_id="d2"), _seed("sim2", doc_id="d3")]
        exp = NeighborExpander(MagicMock(), "lit", vector_retriever=vec)
        out = exp.expand([seed], [EXPAND_SIMILAR])
        self.assertEqual({h.pk for h in out}, {"sim1", "sim2"})
        vec.retrieve.assert_called_once()

    def test_no_vector_retriever_skips(self):
        exp = NeighborExpander(MagicMock(), "lit", vector_retriever=None)
        out = exp.expand([_seed("s1")], [EXPAND_SIMILAR])
        self.assertEqual(out, [])


class TestCapAndTagging(unittest.TestCase):
    def test_max_total_cap(self):
        seed = _seed("s1", doc_id="d1", para=5)
        client = MagicMock()
        client.query.return_value = [_row(f"p{i}", doc_id="d1", para=i) for i in range(20)]
        exp = NeighborExpander(client, "lit", adjacent_window=10, max_total=3)
        out = exp.expand([seed], [EXPAND_ADJACENT])
        self.assertEqual(len(out), 3)

    def test_sources_tagged(self):
        seed = _seed("s1", doc_id="d1", para=5)
        client = MagicMock()
        client.query.return_value = [_row("p4", doc_id="d1", para=4)]
        exp = NeighborExpander(client, "lit")
        out = exp.expand([seed], [EXPAND_ADJACENT])
        self.assertEqual(out[0].sources, [f"{ROUTE_NEIGHBOR}:{EXPAND_ADJACENT}"])
        self.assertEqual(out[0].stage, ROUTE_NEIGHBOR)


class TestApplyNeighborExpansion(unittest.TestCase):
    def test_adds_neighbor_key(self):
        res = {"progressive": LocalRetrieveResult(chunk_hits=[_seed("s1", para=5)])}
        client = MagicMock()
        client.query.return_value = [_row("p4", para=4)]
        exp = NeighborExpander(client, "lit")
        out = apply_neighbor_expansion(res, modes=["adjacent"], expander=exp)
        self.assertIn(ROUTE_NEIGHBOR, out)
        self.assertEqual({h.pk for h in out[ROUTE_NEIGHBOR]}, {"p4"})

    def test_noop_when_no_modes(self):
        res = {"progressive": LocalRetrieveResult(chunk_hits=[_seed("s1")])}
        out = apply_neighbor_expansion(res, modes=[], expander=MagicMock())
        self.assertNotIn(ROUTE_NEIGHBOR, out)
        self.assertIs(out, res)

    def test_noop_when_no_seeds(self):
        out = apply_neighbor_expansion({}, modes=["adjacent"], expander=MagicMock())
        self.assertNotIn(ROUTE_NEIGHBOR, out)


class TestDecisionBuilderExpandMapping(unittest.TestCase):
    """FC plan/metadata args 的 expand 字段 → RouteDecision.expand_neighbors (极简路径)。"""

    def test_progressive_similar(self):
        args = {"paths": [{"t": "progressive", "kw": ["方法"], "expand": ["similar"]}]}
        dec = build_from_plan_args(args, query="其他类似方法")
        self.assertEqual(dec.expand_neighbors, ["similar"])

    def test_metadata_page_assets(self):
        args = {"paths": [{"t": "metadata", "figs": ["3"], "expand": ["page", "assets"]}]}
        dec = build_from_plan_args(args, query="图3附近的文字")
        self.assertEqual(dec.expand_neighbors, ["page", "assets"])

    def test_invalid_expand_dropped(self):
        args = {"paths": [{"t": "progressive", "kw": ["x"], "expand": ["bogus"]}]}
        dec = build_from_plan_args(args, query="x")
        self.assertEqual(dec.expand_neighbors, [])

    def test_no_expand_default_empty(self):
        args = {"paths": [{"t": "progressive", "kw": ["x"]}]}
        dec = build_from_plan_args(args, query="x")
        self.assertEqual(dec.expand_neighbors, [])


if __name__ == "__main__":
    unittest.main()
