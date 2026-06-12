"""从 knowledge_blocks.json 批量生成 RAGAS 评估测试集。

读取 mineru_result 下每篇文献的全量 chunk (knowledge_blocks.json),
喂给 LLM 生成 QA 对, 输出符合 ragas_eval 要求的 test_dataset.json 格式。

用法:
    1. 编辑 config.py, 填写 API_BASE、MODEL、API_KEY
    2. 全量 632 篇已切块文献 (默认):
         python gen_qa_dataset.py
         python gen_qa_dataset.py --list-only   # 只导出待处理清单
    3. 仅 Milvus 在库子集 (与旧 corpus33 行为一致):
         python gen_qa_dataset.py --scope in_corpus --output test_dataset_corpus33.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as user_config
from chunk_resolve import assemble_full_text, build_block_index, resolve_ground_contexts
from corpus_index import doc_title, is_in_corpus

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MINERU_RESULT = (
    Path(user_config.MINERU_RESULT_DIR)
    if user_config.MINERU_RESULT_DIR
    else PROJECT_ROOT / "uploads/kb_v2_b8db35a2"
)
MAX_QA_PER_DOC = 5


def _build_llm_client():
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.clients.llm import LLMClient

    api_base = user_config.API_BASE.strip()
    model = user_config.MODEL.strip()
    api_key = (user_config.API_KEY or os.environ.get("LLM_API_KEY", "")).strip()

    if not api_key:
        raise ValueError("请在 config.py 中设置 API_KEY，或设置环境变量 LLM_API_KEY")
    if not api_base:
        raise ValueError("请在 config.py 中设置 API_BASE")
    if not model:
        raise ValueError("请在 config.py 中设置 MODEL")

    return LLMClient(api_base=api_base, model=model, api_key=api_key, timeout=180, max_retries=3)


def discover_papers(
    skip_dirs: Optional[List[str]] = None,
    *,
    require_vectorized: bool = True,
) -> List[Dict[str, Any]]:
    """扫描 mineru_result 下所有已构建 chunk 的子目录 (knowledge_blocks.json)。"""
    if not MINERU_RESULT.is_dir():
        raise FileNotFoundError(f"文献目录不存在: {MINERU_RESULT}")

    skip = set(skip_dirs or [])
    papers: List[Dict[str, Any]] = []

    for subdir in sorted(MINERU_RESULT.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in skip:
            logger.info(f"跳过目录: {subdir.name}")
            continue

        kb_path = subdir / "knowledge_blocks.json"
        if not kb_path.exists():
            continue
        if require_vectorized and not (subdir / "knowledge_blocks_vec.json").exists():
            logger.debug(f"跳过 (无 knowledge_blocks_vec.json): {subdir.name}")
            continue

        with open(kb_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            continue

        papers.append({
            "doc_name": subdir.name,
            "kb_path": str(kb_path),
            "chunks": chunks,
            "n_chunks": len(chunks),
        })

    return papers


def _resolve_milvus_db() -> Path:
    raw = getattr(user_config, "MILVUS_DB", None)
    return Path(raw) if raw else PROJECT_ROOT / "milvus_lite.db"


def _resolve_in_corpus_only(scope: Optional[str] = None) -> bool:
    """是否只处理 Milvus 在库文献。"""
    key = (scope or getattr(user_config, "DATASET_SCOPE", "") or "").strip().lower()
    if key == "in_corpus":
        return True
    if key == "all":
        return False
    # legacy fallback
    return bool(getattr(user_config, "IN_CORPUS_ONLY", False))


def _annotate_corpus_status(
    papers: List[Dict[str, Any]],
    db_path: Path,
) -> None:
    """为每篇文献标注 in_corpus (是否已灌 Milvus)。"""
    for paper in papers:
        name = paper["doc_name"]
        paper["in_corpus"] = is_in_corpus(name, db_path) if db_path.exists() else False


def filter_papers_for_scope(
    papers: List[Dict[str, Any]],
    *,
    in_corpus_only: bool,
    doc_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = papers
    if doc_filter:
        out = [p for p in out if doc_filter in p["doc_name"]]
    if in_corpus_only:
        out = [p for p in out if p.get("in_corpus")]
    return out


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 mineru_result/knowledge_blocks.json 批量生成 RAG 评测 QA 数据集",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "in_corpus"),
        default=None,
        help="all=全部已切块文献 (默认); in_corpus=仅 Milvus 在库子集",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"输出 JSON 文件名 (默认 config.OUTPUT_FILE={user_config.OUTPUT_FILE})",
    )
    parser.add_argument(
        "--doc-filter",
        default=None,
        help="只处理目录名包含该关键词的文献",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理 N 篇 (0=不限制, 调试用)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只列出将处理的文献清单, 不调用 LLM",
    )
    parser.add_argument(
        "--no-require-vec",
        action="store_true",
        help="不要求 knowledge_blocks_vec.json 存在",
    )
    return parser.parse_args(argv)


def build_qa_system_prompt(doc_title: str = "") -> str:
    title_hint = (
        f"\n\n【本篇文献标题】《{doc_title}》"
        "\n(下面要求把文献名写进问题时, 用这个标题。)"
        if doc_title else ""
    )
    return (
        "你是一名科研文献RAG检索评估数据集生成专家。"
        "你将收到一篇文献的结构化内容（含标题、摘要、正文、图表、表格块），"
        "请据此生成可用于 RAGAS 评估的问答对。"
        f"{title_hint}\n\n"
        "【★最重要: 每条问答必须能被自动评测★】\n"
        "评测方式是: 把问题喂给检索系统, 看它能否召回到 ground_contexts 所在的原文块。"
        "因此你生成的每条都必须标注 eval_kind, 并遵守对应的自包含规则:\n\n"
        "eval_kind = \"fact_retrieval\"  (全库可检索的具体事实)\n"
        "  - 问题问一个具体事实, 答案就在 1-2 个原文块里。\n"
        "  - 【硬性】问题必须自包含: 严禁出现\"这篇/本文/该研究/它/上面那篇\"等指代;"
        "  问题里要带足够具体的实体/术语, 使其在全库范围内也能唯一定位。\n"
        "eval_kind = \"doc_scoped_retrieval\"  (须先锁定到本篇内部的定位题)\n"
        "  - 图/表/页/段落/实体/参考文献等只在本篇内部才有意义的定位题。\n"
        "  - 【硬性】问题里必须写出文献名《标题》(用上面给的标题), 例如"
        "\"《XXX》中的图3说明了什么\", 因为评测会按本篇硬过滤后再检索。\n"
        "eval_kind = \"generation\"  (需跨多块综合生成, 不测检索召回)\n"
        "  - 总结/概述/核心贡献/研究意义/对比等需要综合全文的题。\n"
        "  - 同样必须写出文献名《标题》, 例如\"《XXX》的核心贡献是什么\"。\n"
        "eval_kind = \"skip_auto\"  (不适合自动检索打分)\n"
        "  - 答案是\"有没有/是不是\"(布尔)、\"几篇/多少个\"(计数)、或故意模糊歧义的题。\n\n"
        f"【数量原则 - 每篇最多 {MAX_QA_PER_DOC} 条】\n"
        f"1. 每篇文献最多生成 {MAX_QA_PER_DOC} 条问答；严禁超过 {MAX_QA_PER_DOC} 条。\n"
        "2. 内容只有一小段、信息极少时，只生成 1 条（甚至 0 条）也可以，"
        "不要为了凑数编造问题或答案。\n"
        f"3. 内容丰富时，也只选取最有代表性的 {MAX_QA_PER_DOC} 条以内，避免重复或近义改写。\n"
        "4. 若无法从给定内容中提取任何可靠问答，输出空数组 []。\n"
        "5. 优先多产出 fact_retrieval 与 doc_scoped_retrieval; generation/skip_auto 适量即可。\n\n"
        "【核心原则】\n"
        "1. 每个问答对必须能从提供的文献内容中找到答案，禁止编造。\n"
        "2. 问题必须模拟真实用户发话，自然口语化，不要写成学术考试题。\n"
        "3. ground_truth 参考答案应准确、简洁、直接从文献内容提取。\n"
        "4. ground_contexts 填写 1-3 个最相关 knowledge block 的 id"
        "（正文每块头部的 id=xxx），须从文档中实际出现的 id 选取，禁止编造 id。"
        "内容极少时 1 个 id 即可。\n"
        "5. 每个问答对必须标注 query_type，从下面的类型列表中选取。\n\n"
        "【检索路径类型 - 仅当文献内容支持时才生成，不强制覆盖】\n\n"
        "类型 1: summary - 文献发现/盘存/概览\n"
        '触发条件: 用户想知道"有没有/哪些/是否存在"某类文献，或要求总结/对比/概述。\n'
        "发话示例:\n"
        '  - "有没有关于XXX的研究"\n'
        '  - "总结一下这篇文献的主要内容"\n'
        '  - "这篇论文的核心贡献是什么"\n'
        '  - "概述一下XXX的研究现状"\n'
        'query_type 填: "summary"\n\n'
        "类型 2: progressive - 具体事实/机制/数据问题，未锁定文献\n"
        '触发条件: 用户问"是什么/为什么/如何/多少/怎样"等具体问题，但没有指明文献名。\n'
        "发话示例:\n"
        '  - "XXX的机理是什么"\n'
        '  - "为什么XXX会导致YYY"\n'
        '  - "XXX方法的具体步骤是什么"\n'
        '  - "XXX参数对YYY有什么影响"\n'
        '  - "XXX过程中发生了什么变化"\n'
        'query_type 填: "progressive"\n\n'
        "类型 3: local - 指定文献名或回指某篇文献\n"
        '触发条件: 用户明确提到文献名，或回指"这篇/那篇/第X篇"。\n'
        "发话示例:\n"
        '  - "《XXX》这篇文献里XXX是怎么做的"\n'
        '  - "这篇论文的实验方法是什么" (回指)\n'
        '  - "上面那篇文献的结论是什么" (回指)\n'
        'query_type 填: "local"\n\n'
        "类型 4: metadata - 图表编号/页码/段落/实体精确定位\n"
        "触发条件: 用户提到具体图号、表号、页码、段落号、或要求精确查找某个术语。"
        "文献中没有对应图表/页码/实体时，不要生成此类问题。\n"
        "子类型:\n"
        '  (a) 图表编号直查: "图N说明了什么" / "表N中的数据" / "图Na" (子图)\n'
        '  (b) 页码/段落定位: "第N页讲了什么" / "第N段"\n'
        '  (c) 实体精确查找: "正文里包含XXX的段落" / "XXX出现在哪里"\n'
        "发话示例:\n"
        '  - "图3说明了什么"\n'
        '  - "表2里的数据怎么看"\n'
        '  - "图2a的机理是什么"\n'
        '  - "第5页讲了什么"\n'
        '  - "正文里包含XXX的段落"\n'
        'query_type 填: "metadata_fig" / "metadata_page" / "metadata_entity"\n\n'
        "【附加变体 - 可选，同样仅在内容支持时生成】\n"
        '- 时间过滤: 在问题中加入年份限定，如"2020年以后的XXX研究" (query_type 不变，额外标注 "time_filter")\n'
        '- 复合查询: 一个问题包含两个独立检索意图，如"A文献的方法和B文献的数据" (query_type 填 "multi")\n'
        '- 参考文献意图: 问"这篇论文引用了哪些文献" / "参考文献中有没有XXX" (query_type 填 "references")\n'
        '- 模糊/歧义: 故意用模糊代词，如"这个方法怎么样" / "它和那个有什么区别" (query_type 填 "ambiguous")\n\n'
        "【输出格式】\n"
        "严格输出 JSON 数组格式，不要任何解释或 markdown 围栏，每条必须含 eval_kind:\n"
        "[\n"
        '  {"question": "耐候钢中复合型夹杂物主要由哪两部分组成？", "ground_truth": "...", '
        '"ground_contexts": ["text_xxx"], "query_type": "progressive", "eval_kind": "fact_retrieval"},\n'
        '  {"question": "《XXX》中的图3说明了什么", "ground_truth": "...", '
        '"ground_contexts": ["image_zzz"], "query_type": "metadata_fig", "eval_kind": "doc_scoped_retrieval"},\n'
        '  {"question": "《XXX》的核心贡献是什么", "ground_truth": "...", '
        '"ground_contexts": ["text_aaa"], "query_type": "summary", "eval_kind": "generation"}\n'
        "]\n\n"
        "【重要提醒】\n"
        "- 必须根据文献中实际存在的图号、表号来构造 metadata 类问题。"
        '如果文献中有"图 1"、"图 2"，就问"图1"/"图2"；不要编造不存在的图号。\n'
        '- 内容不足时不要强行覆盖多种 query_type；优先保证 progressive 或 summary 等基础类型。\n'
        "- ground_contexts 中的 id 必须是文档中实际出现的 block id"
        "（如 text_abc123, image_def456, table_ghi789）。"
    )


def build_user_prompt(doc_name: str, full_text: str) -> str:
    max_chars = 30000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[... 文档过长，已截断 ...]"
    return f"文献: {doc_name}\n\n{full_text}"


def parse_qa_response(raw: str) -> tuple[List[Dict[str, Any]], bool]:
    """从 LLM 返回文本中提取 JSON 数组。返回 (items, parsed_ok)。"""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        logger.warning(f"无法从 LLM 输出中提取 JSON 数组: {raw[:200]}")
        return [], False

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}, 原文: {match.group()[:200]}")
        return [], False

    if not isinstance(items, list):
        logger.warning(f"LLM 输出不是 JSON 数组: {match.group()[:200]}")
        return [], False

    results: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = item.get("question", "").strip()
        gt = item.get("ground_truth", "").strip()
        gc = item.get("ground_contexts", [])
        if q and gt:
            qt = item.get("query_type", "progressive")
            results.append({
                "question": q,
                "ground_truth": gt,
                "ground_contexts": gc if isinstance(gc, list) else [str(gc)],
                "query_type": qt,
                "eval_kind": _normalize_eval_kind(item.get("eval_kind"), qt),
            })

    return results, True


_VALID_EVAL_KINDS = {
    "fact_retrieval", "doc_scoped_retrieval", "generation", "skip_auto",
}

# LLM 漏标 eval_kind 时, 按 query_type 兜底推断
_QTYPE_TO_EVAL_KIND = {
    "progressive": "fact_retrieval",
    "local": "doc_scoped_retrieval",
    "metadata_fig": "doc_scoped_retrieval",
    "metadata_page": "doc_scoped_retrieval",
    "metadata_entity": "doc_scoped_retrieval",
    "references": "doc_scoped_retrieval",
    "summary": "generation",
    "multi": "skip_auto",
    "ambiguous": "skip_auto",
}


def _normalize_eval_kind(raw: Any, query_type: str) -> str:
    val = str(raw or "").strip().lower()
    if val in _VALID_EVAL_KINDS:
        return val
    return _QTYPE_TO_EVAL_KIND.get(query_type, "fact_retrieval")


def _clean_qa_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "ground_contexts": item.get("ground_contexts", []),
        "query_type": item.get("query_type", "progressive"),
        "eval_kind": item.get("eval_kind", "fact_retrieval"),
        # 来源与检索库锚点 (新数据集出生即带, 无需再跑 backfill)
        "_source_doc": item.get("_source_doc"),
        "doc_id": item.get("doc_id"),
        "doc_name": item.get("doc_name"),
        "in_corpus": item.get("in_corpus", False),
    }


def _count_doc_items(items: List[Dict[str, Any]], doc_name: str) -> int:
    return sum(1 for item in items if item.get("_source_doc") == doc_name)


def _cap_qa_pairs_for_doc(
    qa_pairs: List[Dict[str, Any]],
    *,
    doc_name: str,
    existing_count: int,
) -> List[Dict[str, Any]]:
    remaining = max(MAX_QA_PER_DOC - existing_count, 0)
    if remaining <= 0:
        logger.warning(f"  {doc_name} 已达到每篇最多 {MAX_QA_PER_DOC} 条，跳过新增")
        return []
    if len(qa_pairs) > remaining:
        logger.warning(
            f"  本篇返回 {len(qa_pairs)} 条 QA，已有 {existing_count} 条，"
            f"按每篇最多 {MAX_QA_PER_DOC} 条截断为 {remaining} 条"
        )
    return qa_pairs[:remaining]


def _progress_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.progress.json")


def _load_processed_docs(output_path: Path) -> set[str]:
    path = _progress_path(output_path)
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(x) for x in data}
    except json.JSONDecodeError:
        logger.warning(f"进度文件无效，将忽略: {path}")
    return set()


def _save_processed_docs(output_path: Path, docs: set[str]) -> None:
    path = _progress_path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(docs), f, ensure_ascii=False, indent=2)


class IncrementalJsonWriter:
    """每生成一条结果即追加写入 JSON 数组文件。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._items: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._items = data
                    logger.info(f"已加载已有结果 {len(self._items)} 条: {path}")
            except json.JSONDecodeError:
                logger.warning(f"输出文件 JSON 无效，将重新写入: {path}")

    @property
    def items(self) -> List[Dict[str, Any]]:
        return self._items

    def append(self, item: Dict[str, Any]) -> None:
        self._items.append(_clean_qa_item(item))
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_cli(argv)
    in_corpus_only = _resolve_in_corpus_only(args.scope)
    output_name = args.output or user_config.OUTPUT_FILE
    output_path = SCRIPT_DIR / output_name
    doc_filter = args.doc_filter if args.doc_filter is not None else user_config.DOC_FILTER
    require_vec = not args.no_require_vec and bool(
        getattr(user_config, "REQUIRE_VECTORIZED", True)
    )
    db_path = _resolve_milvus_db()

    all_papers = discover_papers(
        skip_dirs=user_config.SKIP_DIRS,
        require_vectorized=require_vec,
    )
    _annotate_corpus_status(all_papers, db_path)
    papers = filter_papers_for_scope(
        all_papers,
        in_corpus_only=in_corpus_only,
        doc_filter=doc_filter,
    )
    if args.limit and args.limit > 0:
        papers = papers[: args.limit]

    n_in = sum(1 for p in all_papers if p.get("in_corpus"))
    logger.info(
        f"mineru 已切块文献: {len(all_papers)} 篇 "
        f"(Milvus 在库 {n_in}, 未灌库 {len(all_papers) - n_in}; DB={db_path})"
    )
    logger.info(
        f"本次 scope={'in_corpus' if in_corpus_only else 'all'} "
        f"→ 将处理 {len(papers)} 篇 (数据源: {MINERU_RESULT})"
    )

    if args.list_only:
        manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
        manifest = [
            {
                "doc_name": p["doc_name"],
                "doc_id": p["doc_name"],
                "title": doc_title(p["doc_name"]),
                "in_corpus": p.get("in_corpus", False),
                "n_chunks": p.get("n_chunks", len(p.get("chunks", []))),
            }
            for p in papers
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"\n清单已写入 {manifest_path} ({len(manifest)} 篇)")
        for row in manifest[:5]:
            flag = "✓库" if row["in_corpus"] else "○未灌"
            print(f"  [{flag}] {row['doc_name'][:60]}  chunks={row['n_chunks']}")
        if len(manifest) > 5:
            print(f"  ... 共 {len(manifest)} 篇")
        return

    writer = IncrementalJsonWriter(output_path)

    if not papers:
        logger.error("没有可处理的文献, 请检查 DATASET_SCOPE / DOC_FILTER / Milvus DB 路径")
        return

    llm = _build_llm_client()
    logger.info(f"LLM 已就绪: {llm.model} @ {user_config.API_BASE}")

    processed_docs = _load_processed_docs(output_path)
    if user_config.CONTINUE_FROM:
        cont_path = SCRIPT_DIR / user_config.CONTINUE_FROM
        if cont_path.exists():
            with open(cont_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    doc = item.get("_source_doc", "")
                    if doc:
                        processed_docs.add(doc)
    if processed_docs:
        logger.info(f"续跑：将跳过 {len(processed_docs)} 篇已处理文献")

    for i, paper in enumerate(papers):
        doc_name = paper["doc_name"]

        if doc_name in processed_docs:
            logger.info(f"[{i+1}/{len(papers)}] 跳过 (已有): {doc_name}")
            continue

        chunks = paper["chunks"]
        block_index = build_block_index(chunks)
        full_text = assemble_full_text(chunks)
        title = doc_title(doc_name)
        # doc_id 恒为 mineru 文件夹全名 (= 灌库后的 Milvus doc_id), 与是否在库无关
        doc_id = doc_name
        in_corpus = bool(paper.get("in_corpus"))
        user_msg = build_user_prompt(doc_name, full_text)
        system_prompt = build_qa_system_prompt(title)

        logger.info(
            f"[{i+1}/{len(papers)}] 处理: {doc_name} "
            f"(in_corpus={in_corpus} doc_id={doc_id} "
            f"{len(full_text)} 字符, {len(block_index)} 个可引用块)"
        )

        try:
            resp = llm.chat(
                system=system_prompt,
                user=user_msg,
                temperature=0.3,
                max_tokens=7000,
                disable_thinking=True,
            )
            raw_answer = resp.get("answer", "")
            qa_pairs, parsed_ok = parse_qa_response(raw_answer)

            if not parsed_ok:
                logger.warning(f"  解析失败，原始输出: {raw_answer[:200]}")
            elif not qa_pairs:
                logger.info(f"  文献内容不足或未提取到有效 QA，跳过写入")
                processed_docs.add(doc_name)
                _save_processed_docs(output_path, processed_docs)
            else:
                existing_count = _count_doc_items(writer.items, doc_name)
                qa_pairs = _cap_qa_pairs_for_doc(
                    qa_pairs,
                    doc_name=doc_name,
                    existing_count=existing_count,
                )
                for qa in qa_pairs:
                    qa["_source_doc"] = doc_name
                    qa["doc_id"] = doc_id
                    qa["doc_name"] = doc_name
                    qa["in_corpus"] = in_corpus
                    raw_ids = qa.get("ground_contexts", [])
                    qa["ground_contexts"] = resolve_ground_contexts(raw_ids, block_index)
                    if raw_ids and not qa["ground_contexts"]:
                        logger.warning(f"  ground_contexts 未能从 id 解析出原文: {raw_ids}")
                    writer.append(qa)
                    logger.info(f"  已写入第 {len(writer.items)} 条 -> {output_path}")

                processed_docs.add(doc_name)
                _save_processed_docs(output_path, processed_docs)
                logger.info(f"  本篇生成 {len(qa_pairs)} 条 QA")

        except Exception as e:
            logger.error(f"  LLM 调用失败: {e}")

        time.sleep(1)

    all_qa = writer.items
    logger.info(f"完成！共 {len(all_qa)} 条 QA，保存至 {output_path}")

    print("\n" + "=" * 60)
    print(f"  共 {len(all_qa)} 条评估数据 ({len(papers)} 篇文献)")
    print("=" * 60)
    for qa in all_qa[:3]:
        print(f"  Q: {qa['question'][:60]}...")
        print(f"  A: {qa['ground_truth'][:60]}...")
        print()
    if len(all_qa) > 3:
        print(f"  ... 共 {len(all_qa)} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
