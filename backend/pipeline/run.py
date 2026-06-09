"""CLI 入口: python -m pipeline.run <command> [options]

命令:
  parse        仅运行 PDF 解析 (mineru | uniparser 二选一), 落盘后即返回
  rebuild      从 MinerU 解析结果目录全量重灌: 先清空集合, 再 chunk → embed → store
  append       从 MinerU 解析结果目录增量追加: 不清空集合, 同名 doc_id 会被覆盖
  load-vec     已有 *_vec.json 直接灌入 Milvus, 跳过 parse / chunk / embed
  upload       load-vec 的别名 (同上)
  query        单次查询: retrieve → generate
  chat         多轮对话: 交互式检索 + 生成
  step         执行单个步骤 (高级: parse/chunk/embed/store/retrieve/generate)
  stats        查看 Milvus 集合统计

全局选项:
  --parser {mineru,uniparser}   覆盖 parsing.backend (默认 mineru)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .flows.ingest import IngestResult


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _backend_overrides(args: argparse.Namespace) -> Optional[dict]:
    """把 CLI flag (--milvus-backend / --parser) 翻译成 Pipeline overrides。"""
    overrides: dict = {}
    milvus_backend = getattr(args, "milvus_backend", None)
    if milvus_backend:
        overrides["milvus"] = {"backend": milvus_backend}
    parser_backend = getattr(args, "parser_backend", None)
    if parser_backend:
        overrides["parsing"] = {"backend": parser_backend}
    return overrides or None


def _cmd_parse(args: argparse.Namespace) -> None:
    """parse: 仅运行 PDF 解析 (mineru / uniparser), 不做 chunk/embed/store。"""
    import os
    from .pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))

    backend = pipe.config.parsing.get("backend", "mineru")
    mode_label = f"parse-only (backend={backend})"
    print(f"模式: {mode_label}")
    print(f"输入: {args.path}")
    print("=" * 60)

    if os.path.isdir(args.path):
        results = pipe.parse_directory(
            args.path,
            pattern=args.pattern,
            per_file_timeout=args.timeout,
        )
    elif os.path.isfile(args.path):
        result = pipe.parse(
            [args.path],
            output_dir=args.output_dir,
            parse_timeout=args.timeout,
        )
        results = [result]
    else:
        print(f"路径不存在: {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n共处理 {len(results)} 篇文献")
    success = sum(1 for r in results if r.steps and all(s.success for s in r.steps))
    failed = len(results) - success
    for i, r in enumerate(results, 1):
        print(f"\n--- 文献 {i} ---")
        _print_ingest_result(r)
    print(f"\n{'='*60}")
    print(f"完成: 成功 {success} 篇, 失败 {failed} 篇")
    if backend == "uniparser":
        out_root = pipe.config.uniparser.get("output_dir", "uniparser_result")
        print(
            f"提示: UniParser 解析产物落到 {out_root}/<pdf_stem>/uniparser_result.json, "
            f"chunker 待下回根据该 schema 实现."
        )


def _cmd_rebuild(args: argparse.Namespace) -> None:
    """rebuild: 清空集合后从 MinerU 解析结果目录批量重灌。"""
    _ingest_from_mineru_dir(args, recreate=True)


def _cmd_append(args: argparse.Namespace) -> None:
    """append: 增量追加, 不清空集合; 默认自动跳过已有 doc_id。"""
    _ingest_from_mineru_dir(args, recreate=False)


def _ingest_from_mineru_dir(args: argparse.Namespace, recreate: bool) -> None:
    from .pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))
    skip_existing = not getattr(args, "force", False)
    mode_label = "rebuild (清空后重灌)" if recreate else (
        "append (增量追加, 跳过已有)" if skip_existing else "append (增量追加, 强制重灌)"
    )
    print(f"模式: {mode_label}")
    print(f"源目录: {args.directory}")
    print("=" * 60)

    if recreate:
        results = pipe.rebuild(args.directory)
    else:
        results = pipe.append(args.directory, skip_existing=skip_existing)

    print(f"\n共处理 {len(results)} 篇文献")
    success_count = 0
    failed_count = 0
    for i, r in enumerate(results, 1):
        all_ok = all(s.success for s in r.steps)
        if all_ok:
            success_count += 1
        else:
            failed_count += 1
        print(f"\n--- 文献 {i} ({r.doc_id or '?'}) ---")
        _print_ingest_result(r)

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success_count} 篇, 失败 {failed_count} 篇")

    # 自动打印集合统计
    print(f"\n{'='*60}")
    print("集合统计:")
    print("=" * 60)
    stats = pipe.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _cmd_load_vec(args: argparse.Namespace) -> None:
    """load-vec / upload: 把已经向量化的 *_vec.json 直接灌进 Milvus。"""
    from .pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))

    skip_existing = getattr(args, "skip_existing", False)
    if args.recreate:
        mode_label = "recreate (清空后重灌)"
    elif skip_existing:
        mode_label = "append (跳过 Milvus 中已有 doc_id)"
    else:
        mode_label = "append (同名 doc_id 覆盖)"
    print(f"模式: {mode_label}")
    print(f"输入: {args.path}")
    if args.no_purge:
        print("策略: 不删除同名 doc_id, 直接 upsert")
    print("=" * 60)

    results = pipe.load_vec(
        args.path,
        recreate=args.recreate,
        purge_existing=not args.no_purge,
        skip_existing=skip_existing,
    )

    print(f"\n{'='*60}")
    print(f"完成: 成功灌入 {len(results)} 个文件")
    total_chunks = sum(int(r.get("count", 0) or 0) for r in results)
    print(f"      合计 {total_chunks} 个 chunk")

    # 自动打印集合统计
    print(f"\n{'='*60}\n集合统计:\n{'='*60}")
    stats = pipe.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _cmd_query(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))
    result = pipe.query(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        stream=args.stream,
        output_file=args.output,
        use_agentic=not args.simple,
    )
    if not args.stream and result.answer:
        print("\n" + result.answer)
    if args.output and not args.stream:
        pass  # 已在 query 内部处理


def _chat_read_line(prompt: str) -> Optional[str]:
    """读取用户输入, ESC 退出返回 None。"""
    import sys
    try:
        import termios
        import tty
    except ImportError:
        # 非 Unix 系统回退到普通 input
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    print(prompt, end="", flush=True)
    try:
        tty.setraw(fd)
        line = ""
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # ESC
                print("\r\n")
                return None
            elif ch == "\r" or ch == "\n":
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                break
            elif ch == "\x7f" or ch == "\x08":  # Backspace / Delete
                if line:
                    line = line[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch == "\x03":  # Ctrl+C
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None
            elif ch == "\x04":  # Ctrl+D
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None
            else:
                line += ch
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return line.strip()


def _cmd_chat(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline
    from .flows.query import ChatSession, DEFAULT_MAX_HISTORY_TURNS
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))
    max_turns = int(
        pipe.config.generation.get("max_history_turns", DEFAULT_MAX_HISTORY_TURNS)
    )
    session = ChatSession(max_turns=max_turns)

    print("DP-RAG 对话模式 (按 ESC 退出, 输入 clear 清空历史)")
    print("=" * 50)

    while True:
        query = _chat_read_line("\n> ")
        if query is None:
            print("退出对话")
            break

        if not query:
            continue
        if query.lower() == "clear":
            session = ChatSession(max_turns=max_turns)
            print("已清空对话历史")
            continue

        stream_on = not getattr(args, "no_stream", False)
        result, session = pipe.chat(
            query=query,
            session=session,
            stream=stream_on,
            use_agentic=not args.simple,
        )

        if result.error:
            print(f"\n[错误] {result.error}")
            continue

        if not stream_on and result.answer:
            print(f"\n{result.answer}")


def _cmd_step(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))

    kwargs = {}
    if args.extra:
        for pair in args.extra:
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
                kwargs[k] = v

    result = pipe.run_step(args.step_name, **kwargs)
    if result.success:
        print(f"步骤 {result.step_name} 完成 ({result.elapsed:.2f}s)")
        if result.data:
            summary = {}
            for k, v in result.data.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    summary[k] = v
                elif isinstance(v, dict):
                    summary[k] = f"<dict, {len(v)} keys>"
                elif isinstance(v, list):
                    summary[k] = f"<list, {len(v)} items>"
                else:
                    summary[k] = f"<{type(v).__name__}>"
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"步骤 {result.step_name} 失败: {result.error}", file=sys.stderr)
        sys.exit(1)


def _cmd_stats(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overrides=_backend_overrides(args))
    result = pipe.stats()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _print_ingest_result(result: IngestResult) -> None:
    for step in result.steps:
        status = "OK" if step.success else f"FAIL({step.error})"
        print(f"  {step.step}: {status} ({step.elapsed:.2f}s)")
    if result.total_chunks:
        print(f"  总计: {result.total_chunks} 个知识块")
    if result.doc_id:
        print(f"  doc_id: {result.doc_id}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="DP-RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="自定义配置文件路径 (YAML)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    parser.add_argument(
        "--milvus-backend",
        choices=["lite", "server"],
        default=None,
        help=(
            "Milvus 后端选择 (覆盖 config.milvus.backend): "
            "lite=本地 Milvus Lite 单文件 DB, server=远程 Milvus 服务 (如 docker-compose). "
            "默认沿用配置文件中的 backend 字段."
        ),
    )
    parser.add_argument(
        "--parser",
        dest="parser_backend",
        choices=["mineru", "uniparser"],
        default=None,
        help=(
            "PDF 解析后端 (覆盖 config.parsing.backend): "
            "mineru=https://mineru.net (默认, 输出 content_list_v2.json), "
            "uniparser=https://uniparser.dp.tech (输出 uniparser_result.json, "
            "下游 chunker 待适配, 当前仅落盘解析结果)."
        ),
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    # ── parse ──────────────────────────────────────────────────────────
    p_parse = sub.add_parser(
        "parse",
        help="仅运行 PDF 解析 (mineru / uniparser 二选一), 不做 chunk/embed/store",
        description=(
            "调用 parsing.backend 配置的解析 API (默认 mineru, 可用 --parser "
            "uniparser 切换), 把解析结果落盘后即返回. 适用于:\n"
            "  - 新增 UniParser 支路验收 / 给下游 chunker 喂样例\n"
            "  - 单独跑解析, 不污染向量库\n\n"
            "用法示例:\n"
            "  # 用 UniParser 解析单篇 PDF, 产物落到 uniparser_result/<stem>/\n"
            "  python -m pipeline.run --parser uniparser parse 论文.pdf\n\n"
            "  # 批量扫一个目录\n"
            "  python -m pipeline.run --parser uniparser parse ./pdf/\n\n"
            "  # MinerU 跑单独解析 (不入库)\n"
            "  python -m pipeline.run --parser mineru parse 论文.pdf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_parse.add_argument(
        "path",
        help="PDF 文件路径 / 目录 (递归扫描 *.pdf)",
    )
    p_parse.add_argument(
        "--output-dir",
        default=None,
        help="输出目录 (None 则用 backend 默认: mineru_result/ 或 uniparser_result/)",
    )
    p_parse.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="单文件解析超时秒数 (默认 1800)",
    )
    p_parse.add_argument(
        "--pattern",
        default="*.pdf",
        help="目录模式下的文件匹配 glob, 默认 *.pdf",
    )

    # ── rebuild ────────────────────────────────────────────────────────
    p_rebuild = sub.add_parser(
        "rebuild",
        help="从 MinerU 解析结果目录全量重灌 (先清空集合, 慎用)",
        description=(
            "rebuild 模式: 先 drop 整个 Milvus 集合, 再扫描目录下所有 "
            "*_content_list_v2.json 重新灌入. 之前所有数据会被清空."
        ),
    )
    p_rebuild.add_argument("directory", help="MinerU 解析结果目录 (如 mineru_result/)")

    # ── append ─────────────────────────────────────────────────────────
    p_append = sub.add_parser(
        "append",
        help="从 MinerU 解析结果目录增量追加 (默认自动跳过已有数据)",
        description=(
            "append 模式: 不清空集合, 直接追加. 同名 doc_id (默认是 PDF 文件名"
            "去后缀) 会被覆盖, 其它已灌入的文献保持不变.\n\n"
            "默认自动跳过集合中已存在的 doc_id, 避免重复 chunk/embed/store.\n"
            "加 --force 可强制重灌已有文档."
        ),
    )
    p_append.add_argument("directory", help="MinerU 解析结果目录 (如 mineru_result/)")
    p_append.add_argument(
        "--force",
        action="store_true",
        help="强制重灌已有 doc_id 的文档 (默认自动跳过已存在的文档)",
    )

    # ── load-vec / upload ──────────────────────────────────────────────
    _UPLOAD_HELP = (
        "将已向量化的 *_vec.json 直接灌入 Milvus (跳过 parse/chunk/embed)"
    )
    _UPLOAD_DESC = (
        "跳过解析/分块/向量化, 把已有的 *_vec.json (如 knowledge_blocks_vec.json) "
        "直接 push 进 Milvus. 适用: 本地已跑完 chunk + embedding, 只需上传.\n\n"
        "用法示例 (请在仓库父目录执行, 见下方说明):\n"
        "  # 扫描目录下所有 *_vec.json, 灌进远程 Milvus\n"
        "  python -m pipeline.run --milvus-backend server upload mineru_result/\n\n"
        "  # 增量: 跳过集合里已有的 doc_id\n"
        "  python -m pipeline.run upload mineru_result/ --skip-existing\n\n"
        "  # 单个文件\n"
        "  python -m pipeline.run upload mineru_result/paper1/knowledge_blocks_vec.json"
    )

    def _add_upload_parser(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(
            name,
            help=_UPLOAD_HELP,
            description=_UPLOAD_DESC,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        p.add_argument(
            "path",
            help="目录 (递归扫描 *_vec.json) / glob 模式 / 单个 .json 文件",
        )
        p.add_argument(
            "--recreate",
            action="store_true",
            help="先 drop 整个集合再灌 (慎用, 会清空已有所有文档)",
        )
        p.add_argument(
            "--no-purge",
            action="store_true",
            help="不按 doc_id 删除同名文档, 直接 upsert (默认: 先 purge 再写, 实现覆盖)",
        )
        p.add_argument(
            "--skip-existing",
            action="store_true",
            help="跳过 Milvus 中已存在的 doc_id (增量追加, 不覆盖)",
        )
        return p

    _add_upload_parser("load-vec")
    _add_upload_parser("upload")

    # ── query ──────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="单次查询: retrieve → generate")
    p_query.add_argument("--query", required=True, help="查询问题")
    p_query.add_argument("--mode", default=None, choices=["hybrid", "vector", "metadata"])
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.add_argument("--stream", action="store_true")
    p_query.add_argument("--output", default=None, help="结果输出 JSON 文件")
    p_query.add_argument("--simple", action="store_true", help="使用简单检索模式 (非 Agentic RAG)")

    # ── chat ──────────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="多轮对话: 交互式检索 + 生成 (默认流式输出)")
    p_chat.add_argument("--no-stream", action="store_true", help="关闭流式输出 (默认开启)")
    p_chat.add_argument("--simple", action="store_true", help="使用简单检索模式 (非 Agentic RAG)")

    # ── step ───────────────────────────────────────────────────────────
    p_step = sub.add_parser("step", help="执行单个步骤 (高级)")
    p_step.add_argument("step_name", help="步骤名: parse / chunk / embed / store / retrieve / generate")
    p_step.add_argument("--extra", nargs="*", help="额外参数 key=value 对")

    # ── stats ──────────────────────────────────────────────────────────
    sub.add_parser("stats", help="查看 Milvus 集合统计")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    _setup_logging(args.verbose)

    dispatch = {
        "parse": _cmd_parse,
        "rebuild": _cmd_rebuild,
        "append": _cmd_append,
        "load-vec": _cmd_load_vec,
        "upload": _cmd_load_vec,
        "query": _cmd_query,
        "chat": _cmd_chat,
        "step": _cmd_step,
        "stats": _cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
