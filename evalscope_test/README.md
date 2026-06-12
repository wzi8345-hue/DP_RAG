# DP-RAG 多并发端到端压测 (evalscope)

用 [evalscope perf](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/) 对本项目的对话接口 `POST /api/v1/chat/stream`（默认，对齐前端）做**多并发压测**，关注：

- **首 token 时延 TTFT**（avg / p50 / p99）：首个 `type=text` SSE 事件到达时刻，与界面上答案文字开始展示一致
- 端到端**时延** Latency（`done.latency_s`，avg / p50 / p99 / max）
- **RPS**（每秒请求数）与 token 吞吐
- 不同**并发档位**下的成功率与时延变化

测试用例直接复用 `synthetic_qa_gen/test_dataset_首钢文献.json` 中的 `question`。

> 本项目接口不是 OpenAI 兼容格式，所以这里写了一个自定义 evalscope API 插件
> `dprag_api_plugin.py` 来适配 `{"query": ...}` 请求体与普通 JSON 响应。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `convert_dataset.py` | 把 JSON 数组数据集转成 evalscope `openqa` 用的 `questions.jsonl`（只取 `question`） |
| `dprag_api_plugin.py` | 自定义 API 插件，注册名 `dprag`，适配 `/chat/stream`（流式 TTFT）与 `/query` |
| `run_perf.py` | 压测 runner：扫描多并发档位、落盘结果 |
| `requirements.txt` | 依赖（`evalscope[perf]`） |
| `questions.jsonl` | 由转换脚本生成的数据集（已生成 590 条） |
| `outputs/` | 压测结果（sqlite + summary + HTML 报告），运行后生成 |

## 一次性准备

```bash
cd evalscope_test

# 1) 装依赖（已自带独立 venv，无需污染 .venv-api）
.venv/bin/pip install -r requirements.txt

# 2) 生成数据集（默认读 ../synthetic_qa_gen/test_dataset_首钢文献.json）
.venv/bin/python convert_dataset.py
#   小样本调试: .venv/bin/python convert_dataset.py --limit 30 --dedup
```

## 启动后端

压测前确保 RAG 后端已运行（默认 `http://localhost:8080`）：

```bash
cd ..
bash run_api.sh
```

## 运行压测

```bash
cd evalscope_test

# 默认: 并发档位 1/2/5/10，每档每个 worker 跑 5 条请求
.venv/bin/python run_perf.py

# 自定义并发档位与每档请求数
.venv/bin/python run_perf.py --parallel 1 5 10 20 --requests-per-worker 8

# 走快路径（关闭 agentic）压纯检索+生成时延
.venv/bin/python run_perf.py --no-agentic

# 后端开了鉴权时带上 key
.venv/bin/python run_perf.py --api-key YOUR_KEY
```

### 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--url` | 自动 | 接口地址；默认 `stream` 时用 `/api/v1/chat/stream` |
| `--stream` / `--no-stream` | 开 | 流式压测并单独统计 TTFT；`--no-stream` 走 `/api/v1/query` |
| `--parallel` | `1 2 5 10` | 并发档位列表（空格分隔，逐档压测） |
| `--requests-per-worker` | `5` | 每个并发 worker 的请求数，用于推导每档总请求数 `number` |
| `--number` | 自动 | 直接指定每档总请求数（需与 `--parallel` 等长，覆盖上一项） |
| `--no-agentic` | 关 | 关闭 `use_agentic`，走更快的非智能体路径 |
| `--professional` | 关 | 开启专业研究模式（多轮递进检索，更慢） |
| `--top-k` / `--mode` / `--collection` | 无 | 透传到查询请求体 |
| `--read-timeout` | `600` | 单请求读超时（秒），agentic 较慢时调大 |
| `--api-key` | 无 | 后端 `API_KEYS` 鉴权 |
| `--outputs-dir` | `./outputs` | 结果输出目录 |

完整参数见 `.venv/bin/python run_perf.py -h`。

## 结果在哪看

每次运行输出到 `outputs/<时间戳>/<name>/`：

- 终端直接打印各并发档位的 **TTFT (首 token) / Latency (端到端) / RPS / 吞吐 / 成功率** 汇总表
- `performance_summary.txt`：纯文本汇总
- `perf_report.html`：可视化报告
- sqlite 数据库：每条请求的明细（含 `query_latency`、请求/响应体），可二次分析

> **TTFT 定义（默认流式）**：从发起请求到收到首个 `{"type":"text","content":"..."}` 且 content 非空的时刻。
> `status` / `thinking`（专家模式思考过程）不计入，与前端 ChatView 答案开始展示的时机一致。
> **Latency** 取 `done` 事件的 `latency_s`（服务端统计的端到端耗时）。
> 使用 `--no-stream` 压 `/api/v1/query` 时，TTFT 会退化为端到端时延（无流式首包可区分）。
> token 数若 `done.usage` 有就直接用，否则按中文字数+英文词数粗略估算。

## 备注
- 并发档位之间默认 sleep 5s（`--sleep-interval`）给服务降温，避免连档干扰。
