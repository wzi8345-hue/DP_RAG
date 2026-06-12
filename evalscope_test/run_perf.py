#!/usr/bin/env python3
"""DP-RAG 多并发端到端压测 runner (基于 evalscope perf)。

对 POST /api/v1/chat/stream (默认, 对齐前端) 或 /api/v1/query 做闭环压测,
扫描多个并发档位, 输出每档的:
  - 首 token 时延 TTFT (流式: 首个 type=text 事件, 与界面答案开始展示一致)
  - 端到端时延 Latency (done.latency_s)
  - RPS (每秒请求数) / 吞吐
  - 成功率
结果落盘到 outputs/ (含 sqlite + json 汇总), 可二次分析。

前置:
  1. 后端已启动: bash run_api.sh  (默认 http://localhost:8080)
  2. 已生成数据集: python convert_dataset.py
  3. 装好依赖: .venv/bin/pip install -r requirements.txt

示例:
  # 默认扫描并发 1/2/5/10, 每档 worker 各跑 5 条
  .venv/bin/python run_perf.py

  # 自定义并发档位与每档请求数, 关闭 agentic 走快路径
  .venv/bin/python run_perf.py --parallel 1 5 10 20 --requests-per-worker 8 --no-agentic
"""

from __future__ import annotations

import argparse
from pathlib import Path

# 关键: 导入即注册 @register_api("dprag")
import dprag_api_plugin  # noqa: F401

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

HERE = Path(__file__).parent


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None,
                    help="接口地址 (默认 stream→/chat/stream, 非流式→/query)")
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                    help="流式压测 /chat/stream, 单独统计首 token 时延 (默认开启)")
    ap.add_argument("--api-key", default=None, help="后端 API_KEYS 鉴权 (未开启则不填)")
    ap.add_argument("--dataset-path", default=str(HERE / "questions.jsonl"),
                    help="openqa jsonl 数据集 (含 question 字段)")
    ap.add_argument("--parallel", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="并发档位列表 (空格分隔)")
    ap.add_argument("--requests-per-worker", type=int, default=5,
                    help="每个并发 worker 发送的请求数, 用于推导每档 number")
    ap.add_argument("--number", type=int, nargs="+", default=None,
                    help="每档总请求数 (覆盖 --requests-per-worker, 需与 --parallel 等长)")
    ap.add_argument("--read-timeout", type=int, default=600,
                    help="单请求读超时(秒); RAG agentic 可能较慢")
    ap.add_argument("--connect-timeout", type=int, default=30)
    ap.add_argument("--sleep-interval", type=int, default=5,
                    help="相邻并发档位之间的休眠秒数, 给服务降温")
    ap.add_argument("--name", default="dprag_perf", help="本次压测结果名")
    ap.add_argument("--outputs-dir", default=str(HERE / "outputs"))
    ap.add_argument("--no-test-connection", action="store_true",
                    help="跳过开测前的连通性探测")
    ap.add_argument("--debug", action="store_true")

    # RAG 行为开关 (透传到请求体)
    ap.add_argument("--no-agentic", action="store_true",
                    help="关闭 use_agentic (默认开启)")
    ap.add_argument("--professional", action="store_true",
                    help="开启专业研究模式 (多轮递进检索, 更慢)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--mode", default=None, help="hybrid / vector / metadata")
    ap.add_argument("--collection", default=None, help="目标 Milvus 集合")
    return ap.parse_args()


def main() -> None:
    a = build_args()

    parallel = a.parallel
    if a.number is not None:
        if len(a.number) != len(parallel):
            raise SystemExit("--number 必须与 --parallel 等长")
        number = a.number
    else:
        number = [p * a.requests_per_worker for p in parallel]

    extra_args = {
        "use_agentic": not a.no_agentic,
        "professional": a.professional,
    }
    if a.top_k is not None:
        extra_args["top_k"] = a.top_k
    if a.mode is not None:
        extra_args["mode"] = a.mode
    if a.collection is not None:
        extra_args["collection"] = a.collection

    url = a.url or (
        "http://localhost:8080/api/v1/chat/stream"
        if a.stream
        else "http://localhost:8080/api/v1/query"
    )

    args = Arguments(
        model="dp-rag",
        url=url,
        api="dprag",
        api_key=a.api_key,
        dataset="openqa",
        dataset_path=a.dataset_path,
        parallel=parallel,
        number=number,
        stream=a.stream,
        apply_chat_template=False,
        read_timeout=a.read_timeout,
        connect_timeout=a.connect_timeout,
        sleep_interval=a.sleep_interval,
        no_test_connection=a.no_test_connection,
        name=a.name,
        outputs_dir=a.outputs_dir,
        debug=a.debug,
        extra_args=extra_args,
    )

    print(
        f"[run] url={url} stream={a.stream} parallel={parallel} "
        f"number={number} extra={extra_args}"
    )
    run_perf_benchmark(args)


if __name__ == "__main__":
    main()
