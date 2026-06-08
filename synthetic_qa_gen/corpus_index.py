"""检索库 (Milvus) 在库文献索引: 不依赖 pymilvus, 直接读 milvus-lite 的 sqlite 文件。

关键事实 (来自 pipeline/clients/milvus.py):
  - 灌库时 doc_id 由文件名/文件夹名推断, pk = f"{doc_id}::{chunk_id}"。
  - 对这批 mineru 数据, doc_id 字段的真实值 == mineru 文件夹全名
    (中文标题 + "_" + CNKI 码), 例如
    "(Al,Mg,Ca,Mn)-oxy-sulfide型夹杂物对耐候钢局部腐蚀的诱发研究_FYHS202011001131"。

⚠️ 不要用 "::"" 前的 ASCII 尾巴 (如 _FYHS202011001131) 当 doc_id —— 那是把中文标题
   按 latin-1 误切后的残缺值, 用它做 `doc_id == "..."` 过滤会一条都匹配不上。

判定某 mineru 文件夹是否在库: 该文件夹全名(utf-8) + b"::" 是否作为 pk 前缀出现在库字节里。
doc_id 即取文件夹全名本身。
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

_CHUNK_TYPES = ("title", "text", "table", "image", "summary")


@lru_cache(maxsize=8)
def _corpus_blob(db_path: str) -> bytes:
    """把库里所有 chunk 行的原始字节拼成一块, 供子串命中判定 (缓存, 避免重复读)。"""
    if not Path(db_path).exists():
        return b""
    con = sqlite3.connect(db_path)
    con.text_factory = bytes
    try:
        return b"".join(b for (b,) in con.execute("select data from literature_chunks"))
    finally:
        con.close()


def is_in_corpus(folder_name: str, db_path: Path) -> bool:
    """文件夹全名是否作为 pk 前缀 (doc_id) 真实存在于检索库。"""
    blob = _corpus_blob(str(db_path))
    if not blob:
        return False
    needle = folder_name.encode("utf-8")
    return any(needle + b"::" + t.encode() + b"_" in blob for t in _CHUNK_TYPES)


def dir_suffix(dir_name: str) -> str:
    return dir_name.rsplit("_", 1)[-1]


def doc_title(dir_name: str) -> str:
    """去掉尾部的 _<CNKI码> 得到人类可读标题 (供问题自包含时引用)。"""
    return re.sub(r"_[A-Za-z0-9.]+$", "", dir_name).strip() or dir_name


def corpus_folders(mineru_dir: Path, db_path: Path) -> List[str]:
    """检索库里真实存在的 mineru 文件夹全名列表 (= doc_id 列表)。"""
    out: List[str] = []
    for sub in sorted(Path(mineru_dir).iterdir()):
        if (
            sub.is_dir()
            and (sub / "knowledge_blocks.json").exists()
            and is_in_corpus(sub.name, db_path)
        ):
            out.append(sub.name)
    return out


def build_corpus_json(mineru_dir: Path, db_path: Path) -> Dict[str, Dict[str, str]]:
    """{doc_id(=文件夹全名): {doc_name, title, suffix}} for 在库文献。"""
    out: Dict[str, Dict[str, str]] = {}
    for name in corpus_folders(mineru_dir, db_path):
        out[name] = {"doc_name": name, "title": doc_title(name), "suffix": dir_suffix(name)}
    return out


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve().parent
    root = here.parent
    ap = argparse.ArgumentParser(description="导出在库文献映射 corpus_docs.json")
    ap.add_argument("--mineru-dir", default=str(root / "mineru_result"))
    ap.add_argument("--milvus-db", default=str(root / "milvus_lite.db"))
    ap.add_argument("--output", default=str(here / "corpus_docs.json"))
    args = ap.parse_args()

    mapping = build_corpus_json(Path(args.mineru_dir), Path(args.milvus_db))
    json.dump(mapping, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"在库文献: {len(mapping)} 篇 -> {args.output}")
    for did in list(mapping)[:5]:
        print(f"  doc_id={did}")
