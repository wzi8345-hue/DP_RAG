#!/usr/bin/env python3
"""把 synthetic_qa_gen 的 JSON 数组数据集转成 evalscope `openqa` 需要的 jsonl。

evalscope perf 的 `openqa` 数据集模式在指定 --dataset-path 时, 逐行读取 jsonl
并取每行的 `question` 字段作为压测 prompt。本脚本只抽取 question, 丢弃其余评测
字段 (ground_truth / contexts 等压测用不到)。

用法:
    python convert_dataset.py \
        --src ../synthetic_qa_gen/test_dataset_首钢文献.json \
        --out questions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SRC = "../synthetic_qa_gen/test_dataset_首钢文献.json"
DEFAULT_OUT = "questions.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="源 JSON 数组数据集路径")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 jsonl 路径")
    ap.add_argument("--limit", type=int, default=0,
                    help="只取前 N 条 (0=全部), 调试压测时可先用小样本")
    ap.add_argument("--dedup", action="store_true",
                    help="按 question 文本去重")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = (Path(__file__).parent / src).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = (Path(__file__).parent / out).resolve()

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"源文件不是 JSON 数组: {src}")

    seen: set[str] = set()
    rows: list[str] = []
    for item in data:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        if args.dedup:
            if q in seen:
                continue
            seen.add(q)
        rows.append(json.dumps({"question": q}, ensure_ascii=False))
        if args.limit and len(rows) >= args.limit:
            break

    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[ok] 写入 {len(rows)} 条 question -> {out}")


if __name__ == "__main__":
    main()
