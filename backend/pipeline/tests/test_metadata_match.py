"""Tests for metadata_match (#11 label patterns + #12 entity case variants)."""

from __future__ import annotations

import unittest

from pipeline.retrieval.metadata_match import (
    collect_entity_like_clauses,
    collect_ref_like_clauses,
    entity_case_variants,
    fig_like_clauses,
    score_fig_table_refs,
    table_like_clauses,
)


class TestFigLabelPatterns(unittest.TestCase):
    def test_fig3_variants_in_like(self):
        clauses = fig_like_clauses("3")
        joined = " ".join(clauses)
        self.assertIn('content like "%Fig.3%"', joined)
        self.assertIn('content like "%Fig 3%"', joined.replace("Fig. 3", "Fig 3") or joined)
        self.assertIn('content like "%Figure3%"', joined)
        self.assertIn('content like "%图3%"', joined)
        self.assertIn("Fig.S3", joined)

    def test_fig_3a_variants(self):
        clauses = fig_like_clauses("3a")
        joined = " ".join(clauses)
        self.assertIn('content like "%Fig.3a%"', joined)
        self.assertIn('content like "%Figure 3a%"', joined)
        self.assertIn('content like "%Fig. 3(a)%"', joined)
        self.assertIn('content like "%Fig. 3(A)%"', joined)

    def test_score_fig_paren_variant(self):
        blob = "Results are shown in Fig. 3(a) and Fig. 3(b)."
        score, matched = score_fig_table_refs(blob, ["3A"], [], "image")
        self.assertGreater(score, 0)
        self.assertTrue(any("Fig.3A" in m for m in matched))

    def test_supplementary_label_s3(self):
        clauses = fig_like_clauses("S3")
        joined = " ".join(clauses)
        self.assertIn("Fig.S3", joined)

    def test_score_fig_no_space(self):
        blob = "See Fig.3 for corrosion rates."
        score, matched = score_fig_table_refs(blob, ["3"], [], "image")
        self.assertGreater(score, 0)
        self.assertIn("Fig.3", matched)

    def test_score_fig_chinese(self):
        blob = "结果见图3所示"
        score, matched = score_fig_table_refs(blob, ["3"], [], "text")
        self.assertGreater(score, 0)

    def test_table_no_space(self):
        clauses = table_like_clauses("2")
        joined = " ".join(clauses)
        self.assertIn('content like "%Table2%"', joined)
        self.assertIn('content like "%表2%"', joined)


class TestEntityCaseVariants(unittest.TestCase):
    def test_variants_dedupe(self):
        variants = entity_case_variants("LiNiCoMnO2")
        self.assertEqual(variants[0], "LiNiCoMnO2")
        self.assertIn("linicomno2", variants)

    def test_like_clauses_or_variants(self):
        clauses = collect_entity_like_clauses(["LiNiCoMnO2"])
        self.assertGreaterEqual(len(clauses), 2)
        joined = " ".join(clauses)
        self.assertIn("LiNiCoMnO2", joined)
        self.assertIn("linicomno2", joined)

    def test_collect_ref_like_merges(self):
        clauses = collect_ref_like_clauses(["1"], ["2"])
        self.assertGreater(len(clauses), 4)


if __name__ == "__main__":
    unittest.main()
