"""回填评测数据集: 恢复每条 QA 的源文献与检索库锚点 (纯后处理, 不调用 LLM)。

背景:
    gen_qa_dataset.py 的 _clean_qa_item 丢掉了 _source_doc, 且 ground_contexts
    被解析成了原文文本 (不是 block id)。本脚本用 ground_contexts 文本反查
    mineru_result 下每篇 knowledge_blocks.json 的 block_index, 恢复:
      - _source_doc : QA 来自哪个 mineru 目录
      - doc_id      : 该文献在 Milvus 里的规范 doc_id (协议A下探锚点; 不在库则 None)
      - doc_name    : 人类可读文献名 (= mineru 目录名)
      - in_corpus   : 该文献是否真的已灌进 Milvus 检索库
      - gold_chunk_ids : 反查到的 mineru block id (仅供参考; 注意与 Milvus chunk_id 不一定一致)

    关键事实 (务必知晓): Milvus 检索库可能只覆盖了 mineru 的一个子集。
    in_corpus=False 的 QA 指向的文献根本不在库里, 检索必然失败, 评测时应过滤掉。

用法:
    python backfill_dataset.py \
        --dataset test_dataset_0525.json \
        --output  test_dataset_0525.enriched.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

from chunk_resolve import build_block_index

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Milvus pk 形如 "<doc_id>::<chunk_id>"; chunk_id = mineru block id 形态
_PK_RE = re.compile(
    r"([\x21-\x7e\u4e00-\u9fff]{2,60}?)::"
    r"((?:title|text|table|image|summary)_[a-z0-9]+(?:_p\d+)?)"
)


def _norm(text: str, prefix: int = 100) -> str:
    """归一文本作为反查 key: 去所有空白, 取前 prefix 字符。"""
    return re.sub(r"\s+", "", text or "")[:prefix]


def read_milvus_doc_ids(db_path: Path) -> set[str]:
    """从 milvus-lite 的 sqlite 文件里抽出所有 doc_id (不依赖 pymilvus)。

    milvus-lite 把每行存成二进制 blob (literature_chunks.data), 里面能找到
    "<doc_id>::<chunk_id>" 文本片段, 据此抽 doc_id。
    """
    if not db_path.exists():
        logger.warning(f"Milvus DB 不存在, in_corpus 将全部置 False: {db_path}")
        return set()
    con = sqlite3.connect(str(db_path))
    con.text_factory = bytes
    doc_ids: set[str] = set()
    try:
        for (blob,) in con.execute("select data from literature_chunks"):
            s = blob.decode("latin-1", "ignore")
            for m in _PK_RE.finditer(s):
                doc_ids.add(m.group(1))
    finally:
        con.close()
    return doc_ids


def build_text_lookup(
    mineru_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """遍历所有 mineru 文献, 建两张反查表。

    Returns:
        text2dir : norm(block_text) -> mineru 目录名
        text2bid : norm(block_text) -> mineru block id
    """
    text2dir: Dict[str, str] = {}
    text2bid: Dict[str, str] = {}
    n_dirs = 0
    for sub in sorted(mineru_dir.iterdir()):
        kb = sub / "knowledge_blocks.json"
        if not kb.exists():
            continue
        n_dirs += 1
        try:
            chunks = json.load(open(kb, encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取失败, 跳过 {sub.name}: {e}")
            continue
        idx = build_block_index(chunks)
        for bid, txt in idx.items():
            if not txt or len(txt) < 12:
                continue
            key = _norm(txt)
            # 首次出现优先 (避免被后续文献同前缀块覆盖); 短前缀碰撞概率低
            text2dir.setdefault(key, sub.name)
            text2bid.setdefault(key, bid)
    logger.info(f"已索引 {n_dirs} 篇 mineru 文献, {len(text2dir)} 个唯一文本块")
    return text2dir, text2bid


def resolve_source(
    ground_contexts: List[str],
    text2dir: Dict[str, str],
    text2bid: Dict[str, str],
) -> Tuple[Optional[str], List[str]]:
    """用 ground_contexts 文本反查源文献目录 + mineru block id 列表。"""
    src_votes: Counter = Counter()
    block_ids: List[str] = []
    for gc in ground_contexts or []:
        if not isinstance(gc, str):
            continue
        key = _norm(gc)
        d = text2dir.get(key)
        if d:
            src_votes[d] += 1
        b = text2bid.get(key)
        if b:
            block_ids.append(b)
    src = src_votes.most_common(1)[0][0] if src_votes else None
    return src, block_ids


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="回填评测数据集源文献与检索库锚点")
    parser.add_argument("--dataset", default="test_dataset_0525.json")
    parser.add_argument("--output", default=None, help="默认 <dataset>.enriched.json")
    parser.add_argument("--mineru-dir", default=str(PROJECT_ROOT / "mineru_result"))
    parser.add_argument("--milvus-db", default=str(PROJECT_ROOT / "milvus_lite.db"))
    args = parser.parse_args()

    dataset_path = (SCRIPT_DIR / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_path = (
        Path(args.output)
        if args.output
        else dataset_path.with_name(dataset_path.stem + ".enriched.json")
    )
    mineru_dir = Path(args.mineru_dir)
    milvus_db = Path(args.milvus_db)

    data = json.load(open(dataset_path, encoding="utf-8"))
    logger.info(f"加载 {len(data)} 条 QA: {dataset_path}")

    milvus_doc_ids = read_milvus_doc_ids(milvus_db)
    # suffix (CNKI 码) -> 真实 doc_id; mineru 目录名尾部 token 即 suffix
    suffix2docid = {d.lstrip("_"): d for d in milvus_doc_ids}
    logger.info(f"Milvus 检索库文献数: {len(milvus_doc_ids)}")

    text2dir, text2bid = build_text_lookup(mineru_dir)

    per_type = Counter()
    per_type_in = Counter()
    unmatched = 0
    for qa in data:
        qt = qa.get("query_type", "progressive")
        per_type[qt] += 1
        src, block_ids = resolve_source(qa.get("ground_contexts", []), text2dir, text2bid)
        qa["_source_doc"] = src
        qa["gold_chunk_ids"] = block_ids
        if src is None:
            unmatched += 1
            qa["doc_id"] = None
            qa["doc_name"] = None
            qa["in_corpus"] = False
            continue
        suffix = src.rsplit("_", 1)[-1]
        doc_id = suffix2docid.get(suffix)
        qa["doc_id"] = doc_id
        qa["doc_name"] = src
        qa["in_corpus"] = doc_id is not None
        if qa["in_corpus"]:
            per_type_in[qt] += 1

    json.dump(data, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    total_in = sum(per_type_in.values())
    print("\n" + "=" * 64)
    print(f"  回填完成 -> {output_path}")
    print(f"  总 QA: {len(data)} | 反查到源文献: {len(data) - unmatched} | 未匹配: {unmatched}")
    print(f"  在库 (in_corpus=True) QA: {total_in}")
    print("-" * 64)
    print(f"  {'query_type':<18}{'total':>8}{'in_corpus':>12}")
    for qt in sorted(per_type, key=lambda k: -per_type[k]):
        print(f"  {qt:<18}{per_type[qt]:>8}{per_type_in.get(qt, 0):>12}")
    print("=" * 64)


if __name__ == "__main__":
    main()
