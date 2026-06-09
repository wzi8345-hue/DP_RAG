"""MinerU chunker: reference_list 参考文献提取 + 短段落合并 (仅 mineru 支路)."""

from __future__ import annotations

import json
import os
import unittest

from pipeline.processors.chunker import (
    _build_asset_chunk,
    _detect_summary_sections,
    _extract_entries_from_reference_list,
    _extract_first_sentences,
    _extract_last_sentences,
    _extract_reference_entries,
    _flatten_inline,
    _group_logical_paragraphs,
    _is_bibliography_reference_list,
    _is_metadata_line,
    _is_mineru_reference_list,
    _is_references_section,
    _merge_short_paragraph_groups,
    _normalize_spaced_cjk,
    _normalize_text,
    _should_merge_paragraphs,
    build_knowledge_blocks,
)
from pipeline.steps.chunk import _sanitize_doc_title


def _para(text: str) -> dict:
    return {
        "type": "paragraph",
        "content": {"paragraph_content": [{"type": "text", "content": text}]},
    }


def _reference_list(*texts: str) -> dict:
    """构造 MinerU ``list_type=reference_list`` 块。"""
    return {
        "type": "list",
        "content": {
            "list_type": "reference_list",
            "list_items": [
                {
                    "item_type": "text",
                    "item_content": [{"type": "text", "content": t}],
                }
                for t in texts
            ],
        },
    }


class TestReferencesSection(unittest.TestCase):
    def test_strict_and_variant_titles(self):
        for title in (
            "参考文献",
            "参考文献：",
            "参考文献:",
            "6 参考文献",
            "References",
        ):
            with self.subTest(title=title):
                self.assertTrue(_is_references_section(title))

    def test_non_reference_titles(self):
        for title in ("引言", "1 节点构造措施", "Abstract"):
            with self.subTest(title=title):
                self.assertFalse(_is_references_section(title))


class TestReferenceListDetection(unittest.TestCase):
    def test_mineru_reference_list_marker(self):
        item = _reference_list("[1] Foo[J]. Bar, 2020.")
        self.assertTrue(_is_mineru_reference_list(item))
        self.assertTrue(_is_bibliography_reference_list(
            ["[1] Foo[J]. Bar, 2020."],
        ))

    def test_conclusion_list_not_bibliography(self):
        texts = [
            "1）在江津大气环境中暴晒 1 a 后，Q355NH 腐蚀速率最低。",
            "2）耐候钢锈层中孔隙和裂纹更少。",
        ]
        self.assertFalse(_is_bibliography_reference_list(texts))


class TestReferenceListExtraction(unittest.TestCase):
    def test_one_item_per_entry(self):
        item = _reference_list(
            "［1］ 朱玉琴，段志． 高速公路钢护栏镀锌防腐层检测探究［J］． "
            "公路交通科技( 应用技术版) ，2013( 5) : 281 － 283．",
            "［2］ 刘国栋． 预镀锌铝镁合金护栏在高速公路上的应用［J］． "
            "中外建筑，2019( 9) : 172 － 173．",
        )
        entries = _extract_entries_from_reference_list(item)
        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[0].startswith("［1］"))
        self.assertTrue(entries[1].startswith("［2］"))

    def test_merge_continuation_across_items(self):
        item = _reference_list(
            "[3] 马菱薇, 卢桃丽, 张达威, 等. 耐候钢锈层的稳定化处理[J]. "
            "中国表面工程, 2022, 35(4):151-160.",
            "MA Ling-wei, LU Tao-li, ZHANG Da-wei, et al. Stabilization Treatment "
            "and Growth Mechanism of Rust Layers on Weathering Steel Surface[J]. "
            "China Surface Engineering, 2022, 35(4): 151-160.",
        )
        entries = _extract_entries_from_reference_list(item)
        self.assertEqual(len(entries), 2)
        self.assertIn("马菱薇", entries[0])
        self.assertIn("MA Ling-wei", entries[1])

    def test_merge_split_page_fragment(self):
        item = _reference_list(
            "sion[J]. Journal of Materials Science & Technology, 2020, 39: 190-199.",
            "[9] YU Qiang, DONG Chao-fang, et al. Atmospheric Corrosion[J]. "
            "Journal of Iron and Steel Research, 2016, 23(10): 1061-1070.",
        )
        entries = _extract_entries_from_reference_list(item)
        self.assertEqual(len(entries), 2)
        self.assertIn("Journal of Materials", entries[0])
        self.assertTrue(entries[1].startswith("[9]"))

    def test_paragraph_fallback_still_works(self):
        items = [
            _para("［2］李洪翠．冷轧板冲压橘皮缺陷分析［J］．山东冶金，2013( 1) :3．"),
        ]
        entries = _extract_reference_entries(items)
        self.assertEqual(len(entries), 1)
        self.assertIn("李洪翠", entries[0])


class TestShortParagraphMerge(unittest.TestCase):
    def test_colon_label_merges_with_next(self):
        self.assertTrue(
            _should_merge_paragraphs("退火工艺参数设定：", "保温时间： 1 h。")
        )

    def test_merge_short_groups(self):
        g1 = [_para("再结晶试验结果：")]
        g2 = [_para("退火温度与表面硬度的变化规律见表 。")]
        merged = _merge_short_paragraph_groups([g1, g2])
        self.assertEqual(len(merged), 1)

    def test_keyword_label_does_not_merge_into_body(self):
        """关键词行是自闭合 listing, 不能与后续正文段合并。"""
        prev = "关键词: 高速公路; 海洋气候; 钢护栏; 防腐工艺"
        cur = "护栏是高速公路防护设施中常见的防护措施之一，在公路工程设计过程中"
        self.assertFalse(_should_merge_paragraphs(prev, cur))

    def test_keyword_en_label_does_not_merge_into_body(self):
        prev = "KEY WORDS: weathering steel; atmospheric corrosion"
        cur = "Steel atmospheric corrosion is a complex process..."
        self.assertFalse(_should_merge_paragraphs(prev, cur))

    def test_abstract_label_still_merges_with_content(self):
        """摘要行后通常直接接摘要正文 (同段或下一段都应合并), 不属于 listing."""
        # 摘要内容是另一段时, 期望合并到一起
        prev = "摘要"
        cur = "目的 研究3 种典型耐候钢..."
        self.assertTrue(_should_merge_paragraphs(prev, cur))


class TestMetadataLineFilter(unittest.TestCase):
    def test_obvious_metadata_lines(self):
        for txt in (
            "中图分类号：TG172.3",
            "文献标识码：A",
            "文 献 标 志 码 ：B",
            "文章编号：1672-9242(2023)08-0114-08",
            "DOI：10.7643/issn.1672-9242.2023.08.015",
            "doi: 10.1016/j.corsci.2014.01.009",
            "收稿日期：2023-05-12",
            "基金项目：国家自然科学基金（U2003122）",
            "作者简介：高立军（1980—），男，博士",
            "通讯作者：张涛，副研究员",
            "通信作者：李学涛",
            "CLC number: TG172.3",
        ):
            with self.subTest(txt=txt):
                self.assertTrue(_is_metadata_line(txt), msg=txt)

    def test_not_metadata_lines(self):
        for txt in (
            "本文 DOI 申请由出版社统一处理。",
            "耐候钢的腐蚀机理研究",
            "1.1 材料",
            "试验材料为 Q355NH 钢",
            "",
        ):
            with self.subTest(txt=txt):
                self.assertFalse(_is_metadata_line(txt), msg=txt)

    def test_chunker_drops_metadata_paragraphs(self):
        """metadata 行不应进入 chunks (从源头被 _group_logical_paragraphs drop)。"""
        items = [
            _para("关键词：耐候钢；大气腐蚀；江津"),
            _para("中图分类号：TG172.3"),
            _para("文献标识码：A"),
            _para("文章编号：1672-9242(2023)08-0114-08"),
            _para("DOI：10.7643/issn.1672-9242.2023.08.015"),
            _para("本文研究耐候钢的腐蚀行为。"),
        ]
        groups = _group_logical_paragraphs(items)
        # 关键词段保留, 其它 metadata 全 drop, 加上一个正文段 -> 2 组
        flat_text = "".join(
            "".join(p.get("content", {}).get("paragraph_content", [{}])[0].get("content", "")
                    for p in g)
            for g in groups
        )
        self.assertNotIn("中图分类号", flat_text)
        self.assertNotIn("文献标识码", flat_text)
        self.assertNotIn("文章编号", flat_text)
        self.assertNotIn("DOI", flat_text)
        self.assertIn("耐候钢的腐蚀行为", flat_text)


class TestSpacedCJKNormalization(unittest.TestCase):
    def test_collapse_cjk_spacing(self):
        self.assertEqual(
            _normalize_spaced_cjk("可 以 看 到 ， 耐 候 钢 和 碳 钢"),
            "可以看到，耐候钢和碳钢",
        )

    def test_preserve_normal_chinese(self):
        self.assertEqual(_normalize_spaced_cjk("耐候钢和碳钢"), "耐候钢和碳钢")

    def test_preserve_cjk_digit_mix(self):
        # "图 1 所示" — 中间夹了数字, 不该被折叠
        self.assertEqual(_normalize_spaced_cjk("图 1 所示"), "图 1 所示")

    def test_preserve_short_input(self):
        self.assertEqual(_normalize_spaced_cjk("耐 候"), "耐 候")

    def test_normalize_text_combined(self):
        # 既有 CJK 间距, 也有 ASCII 字间距
        s = "A B S T R A C T : 可 以 看 到"
        self.assertEqual(_normalize_text(s), "ABSTRACT : 可以看到")

    def test_collapse_spaced_summary_label(self):
        for raw in ("摘 要", "摘  要", "摘\u3000要", "摘\u200b要", " 摘 要 "):
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_text(raw), "摘要")

    def test_collapse_spaced_summary_label_in_sentence(self):
        s = "摘 要：本文研究了耐候钢在大气环境中的腐蚀行为。"
        self.assertEqual(
            _normalize_text(s),
            "摘要：本文研究了耐候钢在大气环境中的腐蚀行为。",
        )

    def test_flatten_inline_applies_normalization(self):
        items = [
            {"type": "text", "content": "可 以 看 到 ， 耐 候 钢 和 碳 钢 的 锈 层"},
        ]
        self.assertEqual(_flatten_inline(items), "可以看到，耐候钢和碳钢的锈层")

    def test_flatten_inline_preserves_latex(self):
        items = [
            {"type": "text", "content": "化学组成"},
            {"type": "equation_inline", "content": "E = m c^2"},
        ]
        # equation 被 $...$ 包住, 里面的 m c^2 不应被字间距折叠 (实际上也不满足 ≥3 letter+space)
        out = _flatten_inline(items)
        self.assertIn("$E = m c^2$", out)


class TestSanitizeDocTitle(unittest.TestCase):
    def test_strip_cnki_suffix(self):
        cases = [
            ("典型耐候钢在江津大气环境中暴晒1 a的腐蚀行为_JSCX202308015",
             "典型耐候钢在江津大气环境中暴晒1 a的腐蚀行为"),
            ("锌铝镁合金镀层板研究开发_SNAD000001828487", "锌铝镁合金镀层板研究开发"),
            ("防硫酸腐蚀的科技成果_LSGY198302020", "防硫酸腐蚀的科技成果"),
            ("研判市场走势 探寻技术动向_CCPB201103180071", "研判市场走势 探寻技术动向"),
            ("NH35q耐大气腐蚀钢冷裂敏感性试验研究_HSJJ601.000",
             "NH35q耐大气腐蚀钢冷裂敏感性试验研究"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_sanitize_doc_title(raw), expected)

    def test_no_suffix_preserved(self):
        for s in ("普通中文标题", "Title without ID", "钢铁工艺研究"):
            self.assertEqual(_sanitize_doc_title(s), s)

    def test_empty(self):
        self.assertEqual(_sanitize_doc_title(""), "")


class TestAssetChunkPathField(unittest.TestCase):
    """P0-3: image / table chunk 的绝对路径应该在 content 之外的字段。"""

    def test_image_chunk_has_separate_path_field(self):
        item = {
            "type": "image",
            "content": {
                "image_caption": [{"type": "text", "content": "图1 装置示意图"}],
                "image_footnote": [],
                "image_source": {"path": "images/abcd1234.jpg"},
            },
        }
        chunk = _build_asset_chunk(item, page_idx=0, section="1 引言",
                                   images_root="/abs/root")
        self.assertIsNotNone(chunk)
        # 绝对路径放专门字段
        self.assertEqual(chunk["image_path"], "/abs/root/images/abcd1234.jpg")
        # content 里只放相对路径, 不应再含绝对路径
        self.assertIn("[Image Path] images/abcd1234.jpg", chunk["content"])
        self.assertNotIn("/abs/root", chunk["content"])

    def test_table_chunk_has_separate_path_field(self):
        item = {
            "type": "table",
            "content": {
                "table_caption": [{"type": "text", "content": "表1 化学成分"}],
                "table_footnote": [],
                "image_source": {"path": "images/xyz.jpg"},
                "html": "<table><tr><td>A</td></tr></table>",
            },
        }
        chunk = _build_asset_chunk(item, page_idx=0, section="2 方法",
                                   images_root="/some/root")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["table_image_path"], "/some/root/images/xyz.jpg")
        self.assertNotIn("/some/root", chunk["content"])
        self.assertIn("[Table HTML]", chunk["content"])

    def test_image_without_path_omits_field(self):
        item = {
            "type": "image",
            "content": {
                "image_caption": [{"type": "text", "content": "图无路径"}],
                "image_footnote": [],
                "image_source": {"path": ""},
            },
        }
        chunk = _build_asset_chunk(item, page_idx=0, section="", images_root="")
        self.assertIsNotNone(chunk)
        self.assertNotIn("image_path", chunk)


class TestAssetTextLinking(unittest.TestCase):
    """A+B: section 内文本块 ↔ 图表块双向关联 (供检索端 asset 互补扩展)。"""

    def _title(self, text: str) -> dict:
        return {
            "type": "title",
            "content": {"title_content": [{"type": "text", "content": text}]},
        }

    def _image(self, caption: str, path: str = "images/fig1.jpg") -> dict:
        return {
            "type": "image",
            "content": {
                "image_caption": [{"type": "text", "content": caption}],
                "image_footnote": [],
                "image_source": {"path": path},
            },
        }

    def _build(self):
        data = [[
            self._title("2 结果与讨论"),
            _para(
                "由图1可见，GI 板与 ZM 板的镀层表面形貌存在显著差异，"
                "ZM 板因合金相的存在形成了更致密的多相结构，耐蚀性更优。"
            ),
            _para(
                "进一步分析表明，镀层厚度与显微硬度增大时石击坑深度减小，"
                "因此镀层厚且硬度高的钢板抗石击性能更优异。"
            ),
            self._image("图1 不同试样镀层的表面微观形貌"),
        ]]
        return build_knowledge_blocks(
            data, images_root="", doc_title="t",
            summary_enabled=False, summary_llm_enabled=False,
            summary_bm25_enabled=False, summary_embedding_enabled=False,
        )

    def test_text_chunk_links_to_section_image(self):
        chunks = self._build()
        texts = [c for c in chunks if c["type"] == "text"]
        images = [c for c in chunks if c["type"] == "image"]
        self.assertTrue(texts and images)
        img_id = images[0]["id"]
        # section 内所有 text 块都应挂上本节图片 (section 锚点链接)
        for tc in texts:
            ids = {a.get("chunk_id") for a in tc.get("related_assets", [])}
            self.assertIn(img_id, ids, f"text chunk {tc['id']} 未关联到图片")

    def test_image_chunk_links_back_to_text(self):
        chunks = self._build()
        texts = [c for c in chunks if c["type"] == "text"]
        images = [c for c in chunks if c["type"] == "image"]
        text_ids = {c["id"] for c in texts}
        related_ids = {a.get("chunk_id") for a in images[0].get("related_assets", [])}
        # 图片块反向挂上 section 锚点正文块
        self.assertTrue(related_ids & text_ids, "图片块未反向关联到正文")

    def test_related_assets_dedup_no_self_ref(self):
        chunks = self._build()
        for c in chunks:
            ids = [a.get("chunk_id") for a in c.get("related_assets", [])]
            self.assertEqual(len(ids), len(set(ids)), "related_assets 出现重复")
            self.assertNotIn(c["id"], ids, "related_assets 不应自引用")


class TestSentenceExtraction(unittest.TestCase):
    def test_first_sentence_cn(self):
        s = "本文研究 X。然后 Y。最后 Z。"
        self.assertEqual(_extract_first_sentences(s, 1), "本文研究 X。")

    def test_last_sentence_cn(self):
        s = "本文研究 X。然后 Y。最后 Z。"
        self.assertEqual(_extract_last_sentences(s, 1), "最后 Z。")

    def test_short_text_no_terminator(self):
        # 没有句末标点的整段, 整段就是一句
        self.assertEqual(_extract_first_sentences("一句话不带标点", 1), "一句话不带标点")


class TestSummarySectionDetection(unittest.TestCase):
    def _title(self, text: str) -> dict:
        return {
            "type": "title",
            "content": {"title_content": [{"type": "text", "content": text}]},
        }

    def test_spaced_abstract_title_detected_as_summary_section(self):
        for label in ("摘 要", "摘\u200b要", "摘  要"):
            with self.subTest(label=label):
                data = [[
                    self._title(label),
                    _para("本文研究了耐候钢在大气环境中的腐蚀行为，采用失重法测量腐蚀速率。"),
                    self._title("1 引言"),
                    _para("引言正文内容足够长，不会被合并。"),
                ]]
                detection = _detect_summary_sections(
                    data, bm25_enabled=False, embedding_enabled=False,
                )
                self.assertEqual(detection["strategy"], "rule")
                self.assertIn("摘要", detection["summary_sections"])

                chunks = build_knowledge_blocks(
                    data,
                    images_root="",
                    doc_title="t",
                    summary_enabled=True,
                    summary_llm_enabled=False,
                    summary_bm25_enabled=False,
                    summary_embedding_enabled=False,
                )
                summaries = [c for c in chunks if c.get("type") == "summary"]
                self.assertEqual(len(summaries), 1)
                self.assertIn("耐候钢", summaries[0]["content"])


class TestEquationContextInjection(unittest.TestCase):
    """P1-6: equation chunk 应该自动注入前后段的 anchor 句。"""

    def test_equation_gets_neighbor_anchor_in_context(self):
        # 1 个 section 内: 段落 A + 公式 + 段落 B (含 "式中: ...")
        # 期望: equation 的 context 包含 "式中..." 和 "对试样进行除锈..."
        data = [[
            {"type": "title",
             "content": {"title_content": [{"type": "text", "content": "1.2 方法"}], "level": 1}},
            _para("对试样进行除锈,500 mL盐酸+500 mL蒸馏水。"),
            {"type": "equation_interline",
             "content": {"math_content": "D = \\Delta m / \\rho S"}},
            _para("式中:Δm 为腐蚀质量损失;ρ 为试样密度;S为腐蚀面积;D为腐蚀减薄量。"),
        ]]
        chunks = build_knowledge_blocks(
            data, images_root="", doc_title="t",
            summary_enabled=False,
        )
        eq_chunks = [c for c in chunks if c.get("type") == "equation"]
        self.assertEqual(len(eq_chunks), 1)
        ctx = eq_chunks[0].get("context", "")
        self.assertIn("式中", ctx)
        self.assertIn("腐蚀减薄量", ctx)


class TestPreambleParagraphIndex(unittest.TestCase):
    """P1-5: section='' 阶段的 chunk 不消耗 paragraph_index。"""

    def test_orphan_text_does_not_consume_index(self):
        # 前 2 个 paragraph 没有 title; 第 3 项 title 之后才是正文.
        # 期望:
        #   - 前 2 个 orphan chunk 的 paragraph_index = -1, is_preamble=True
        #   - title 之后的第 1 个段落 paragraph_index = 1, 不被 preamble 偏移
        # 各段都用 ≥ 50 字符的长内容, 避免被 _merge_short_paragraph_groups 合并.
        data = [[
            _para("上一篇文章的结尾续文 A, 这一段是不属于目标文献的杂项内容, 应当被识别为 preamble。"),
            _para("上一篇文章的结尾续文 B, 同样也是与目标文献无关的前置噪声段, 期待被打 preamble 标。"),
            {"type": "title",
             "content": {"title_content": [{"type": "text", "content": "目标文献正文标题"}], "level": 1}},
            _para("这才是目标文献的第一段, 应当 paragraph_index = 1, 段落足够长不会被合并。"),
            _para("这是目标文献的第二段, 应当 paragraph_index = 2, 同样保留足够字符以避免被合并。"),
        ]]
        chunks = build_knowledge_blocks(
            data, images_root="", doc_title="t",
            summary_enabled=False,
        )
        text_chunks = [c for c in chunks if c.get("type") == "text"]
        preamble = [c for c in text_chunks if c.get("is_preamble")]
        real_body = [c for c in text_chunks if not c.get("is_preamble")]
        # 前 2 个 orphan 应该 paragraph_index 都为 -1, 且都标 is_preamble
        self.assertEqual(len(preamble), 2)
        for c in preamble:
            self.assertEqual(c["paragraph_index"], -1)
        # 真正属于目标文献的段落, paragraph_index 应从 1 开始连续
        self.assertEqual([c["paragraph_index"] for c in real_body], [1, 2])


class TestPageNumberIsNoise(unittest.TestCase):
    """P2-9: page_number 应该和 page_header/footer 一样被丢弃。"""

    def test_page_number_dropped(self):
        data = [[
            {"type": "page_number",
             "content": {"page_number_content": [{"type": "text", "content": "55"}]}},
            _para("正文第一段。"),
        ]]
        chunks = build_knowledge_blocks(
            data, images_root="", doc_title="t",
            summary_enabled=False,
        )
        joined = "".join((c.get("content") or "") for c in chunks)
        self.assertNotIn("55 ", joined)
        # 至少要有正文段, 验证不是把全部都吞了
        self.assertTrue(any("正文第一段" in (c.get("content") or "") for c in chunks))


class TestBuildOnRealMineruSample(unittest.TestCase):
    _REF_LIST_SAMPLE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "mineru_result",
        "锌铝镁钢护栏在2020年鹤大高速公路波形梁钢护栏改造工程中的应用_LNJT202109018",
        "锌铝镁钢护栏在2020年鹤大高速公路波形梁钢护栏改造工程中的应用_LNJT202109018",
        "f3b29e2f-f59d-43ce-a4df-882da7ef803e_content_list_v2.json",
    )

    def test_reference_list_file_produces_refs(self):
        if not os.path.isfile(self._REF_LIST_SAMPLE):
            self.skipTest(f"sample not found: {self._REF_LIST_SAMPLE}")
        with open(self._REF_LIST_SAMPLE, encoding="utf-8") as f:
            data = json.load(f)
        blocks = build_knowledge_blocks(
            data,
            images_root=os.path.dirname(self._REF_LIST_SAMPLE),
            summary_enabled=False,
            references_batch_size=5,
        )
        ref_blocks = [b for b in blocks if b.get("type") == "references"]
        self.assertGreater(len(ref_blocks), 0)
        joined = "\n".join(b["content"] for b in ref_blocks)
        self.assertIn("［1］", joined)
        self.assertIn("朱玉琴", joined)
        self.assertNotIn("1）在江津", joined)


if __name__ == "__main__":
    unittest.main()
