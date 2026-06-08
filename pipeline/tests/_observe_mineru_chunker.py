"""一次性观察脚本: 对 mineru_result 下的若干 content_list_v2.json
跑 build_knowledge_blocks (关掉 embedder / LLM), 把每个 chunk 的关键字段
打印出来, 用来快速发现 chunk 边界、type 分布、参考文献聚合等问题。

运行: python -m pipeline.tests._observe_mineru_chunker

这是 review/observation 用的一次性脚本, 不是 unittest test case (文件名以 _ 开头,
unittest discover 默认 skip).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

from pipeline.processors.chunker import build_knowledge_blocks
from pipeline.steps.chunk import _sanitize_doc_title

MINERU_ROOT = os.path.expanduser(
    os.environ.get(
        "MINERU_ROOT",
        "/Users/dp/Desktop/工作文件/DP_rag_skill/mineru_result",
    )
)

# 选 3 个代表性样本:
# - 标准学术论文 (中英双摘要 + reference_list + 公式 + 图表)
# - 多文章共一页 PDF (preamble 污染 + cur_section 切换)
# - 工程类文献 (短段落多 + 含参考文献条目)
SAMPLES = [
    "典型耐候钢在江津大气环境中暴晒1 a的腐蚀行为_JSCX202308015",
    "防硫酸腐蚀的科技成果_LSGY198302020",
    "锌铝镁钢护栏在2020年鹤大高速公路波形梁钢护栏改造工程中的应用_LNJT202109018",
]


def _find_v2(stem: str) -> Optional[str]:
    """在 mineru_result/<stem>/<stem>/ 下找 *_content_list_v2.json."""
    sub = os.path.join(MINERU_ROOT, stem, stem)
    if not os.path.isdir(sub):
        return None
    for name in os.listdir(sub):
        if name.endswith("_content_list_v2.json") or name == "content_list_v2.json":
            return os.path.join(sub, name)
    return None


def _content_snippet(c: Dict[str, Any], n: int = 60) -> str:
    s = (c.get("content") or "").replace("\n", " ⏎ ")
    return s[:n] + ("…" if len(s) > n else "")


def _context_snippet(c: Dict[str, Any], n: int = 50) -> str:
    s = (c.get("context") or "").replace("\n", " ⏎ ")
    if not s:
        return ""
    return s[:n] + ("…" if len(s) > n else "")


def _section_snippet(c: Dict[str, Any], n: int = 40) -> str:
    s = (c.get("section") or "").replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def _print_summary(blocks: List[Dict[str, Any]]) -> None:
    types = Counter(b.get("type") for b in blocks)
    preamble_count = sum(1 for b in blocks if b.get("is_preamble"))
    print(f"  总 chunk 数: {len(blocks)}  type 分布: {dict(types)}  preamble: {preamble_count}")
    text_lens = [len(b.get("content") or "")
                 for b in blocks if b.get("type") in ("text", "summary")]
    if text_lens:
        print(
            f"  text/summary 字符数 min={min(text_lens)} "
            f"avg={sum(text_lens) // len(text_lens)} max={max(text_lens)} "
            f"(< 50: {sum(1 for x in text_lens if x < 50)}, "
            f"50-200: {sum(1 for x in text_lens if 50 <= x < 200)}, "
            f"200-800: {sum(1 for x in text_lens if 200 <= x < 800)}, "
            f"800-2000: {sum(1 for x in text_lens if 800 <= x < 2000)}, "
            f"≥2000: {sum(1 for x in text_lens if x >= 2000)})"
        )


def _print_chunks(blocks: List[Dict[str, Any]]) -> None:
    for i, b in enumerate(blocks):
        t = b.get("type", "?")
        pi = b.get("paragraph_index", "?")
        pages = b.get("pages") or []
        page0 = pages[0] + 1 if pages else "?"
        clen = len(b.get("content") or "")
        sec = _section_snippet(b)
        cnt = _content_snippet(b)
        extras: List[str] = []
        if b.get("is_preamble"):
            extras.append("PREAMBLE")
        if b.get("parent_chunk_id"):
            extras.append(
                f"split={b.get('chunk_index')}/{b.get('chunk_total')}")
        if b.get("synthesized"):
            extras.append("LLM-synth")
        if b.get("ref_count"):
            extras.append(f"refs={b.get('ref_count')}")
        if b.get("image_path"):
            extras.append("img_path")
        if b.get("table_image_path"):
            extras.append("tbl_path")
        ra = b.get("related_assets") or []
        if ra:
            extras.append(f"links={len(ra)}")
        flag = " ".join(extras)
        print(
            f"  [{i:>3}] t={t:<10} pg={page0:<3} para={pi:<4} "
            f"len={clen:<5} sec={sec!r:<42} {flag}\n        {cnt!r}"
        )
        if t == "equation":
            ctx = _context_snippet(b, n=120)
            if ctx:
                print(f"        ↳ context: {ctx!r}")


def main() -> int:
    if not os.path.isdir(MINERU_ROOT):
        print(f"MINERU_ROOT 不存在: {MINERU_ROOT}")
        return 2

    for stem in SAMPLES:
        path = _find_v2(stem)
        clean_title = _sanitize_doc_title(stem)
        print(f"\n================ {stem} ================")
        print(f"  → doc_title (sanitized): {clean_title!r}")
        if not path:
            print(f"  跳过: 找不到 v2 json")
            continue
        print(f"  file: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 不传 embedder / llm → 跳过 tier3 + tier4, 只看规则/BM25/promote 行为
        blocks = build_knowledge_blocks(
            data,
            images_root=os.path.dirname(path),
            doc_title=clean_title,
            embedder=None,
            llm=None,
            summary_enabled=True,
            summary_llm_enabled=False,
            summary_embedding_enabled=False,
            references_batch_size=5,
        )
        _print_summary(blocks)
        _print_chunks(blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
