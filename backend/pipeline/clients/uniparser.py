"""UniParser API 客户端: 通过 https://uniparser.dp.tech 解析 PDF。

UniParser 是 MinerU 之外的另一条 PDF 解析支路, 输出 schema 与 MinerU 的
``content_list_v2.json`` 不同 (这里产物保存为 ``uniparser_result.json``,
含 ``content`` / ``objects`` / ``pages_dict`` / ``pages_tree``), 因此后续
chunker 需要基于这个新 schema 单独适配 (下一回任务再做).

调用流程:
1) POST ``/trigger-file-async`` (multipart) 上传 PDF, 提交时即指定唯一
   ``token`` (UUID). 默认 ``sync=true`` 等待解析完成.
2) POST ``/get-formatted`` (JSON) 拿格式化结果, 一次性把
   content / objects / pages_dict / pages_tree 全开, 方便下游灵活选用.
3) 把整个 response 落盘到 ``<output_dir>/<pdf_stem>/uniparser_result.json``,
   同时把 sidecar 元数据 (token / submit_response / parse_mode) 写到
   ``<output_dir>/<pdf_stem>/uniparser_meta.json``, 方便排查.

异步模式 (``sync=false``) 也支持, 此时会反复调 ``/get-formatted`` 直到拿到
非空结果或超过 ``poll_max_retries``; 期间任何 4xx 即视为失败.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ── 解析阶段 (SemanticType -> ParseMode) ──────────────────────────────
# 取值见 https://uniparser.dp.tech/api 文档 "解析配置 SemanticType":
#   0=禁用, 1=OCRFast, 2=OCRHighQuality, -1=DumpBase64, 3=DigitalExported(仅 textual)
# chart / expression / molecule 目前无 high-quality 模式 (传 2 会被拒).
DEFAULT_PARSE_MODES: Dict[str, str] = {
    "textual": "2",
    "equation": "2",
    "table": "2",
    "chart": "-1",
    "figure": "-1",
    "expression": "-1",
    "molecule": "1",
}

# ── 结果获取阶段 (FormatFlag + 输出开关) ──────────────────────────────
# 默认把所有结构化字段全开, 这样下游 chunker 可以挑想用的, 不必再次调 API.
DEFAULT_OUTPUT_FLAGS: Dict[str, Any] = {
    "content": True,
    "objects": True,
    "pages_dict": True,
    "pages_tree": True,
    "molecule_source": False,
    "marginalia": True,
}

# Format: plain / markup / markdown / latex / html
DEFAULT_FORMAT_FLAGS: Dict[str, str] = {
    "textual": "markdown",
    "table": "markdown",
    "equation": "markdown",
    "molecule": "markdown",
    "chart": "markdown",
    "figure": "markdown",
    "expression": "markdown",
}


def _safe_resp_text(resp: "requests.Response", limit: int = 300) -> str:
    """按 UTF-8 解码响应体, 失败时回退到 latin-1, 避免中文报错乱码。"""
    try:
        text = resp.content.decode("utf-8", errors="replace")
    except Exception:
        text = resp.text
    return text[:limit]


_TOKEN_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _make_token(file_path: str) -> str:
    """为每个文件生成符合服务端 ``Token_Invalid`` 校验的唯一 token。

    UniParser 文档明确: token 只允许 ASCII 字母 / 数字 / 下划线 / 短横线;
    不要用 Python 的 ``str.isalnum()`` (Unicode-aware, 中文也算 alnum) 做过滤,
    否则中文 PDF 文件名会让服务端 400 ``Token is invalid``.

    格式: ``<safe_stem>_<uuid8>``; 若 stem 过滤后为空 (例如纯中文标题) 就退化为
    ``doc_<uuid8>``, 始终保证 token 非空且合法.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    safe_stem = _TOKEN_ALLOWED_RE.sub("_", stem).strip("_-")
    if not safe_stem:
        safe_stem = "doc"
    # 截断防止 token 太长 (官方未明确长度限制, 这里保守 < 64)
    if len(safe_stem) > 40:
        safe_stem = safe_stem[:40]
    return f"{safe_stem}_{uuid.uuid4().hex[:8]}"


class UniParserClient:
    """UniParser API (https://uniparser.dp.tech) 客户端。

    与 ``MinerUClient`` 类似, 提供 ``process(file_paths, timeout=None)`` 入口,
    返回 ``{"output_dir": str, "results": [...], "tokens": [...]}``. 每篇 PDF
    的解析产物落到 ``output_dir/<pdf_stem>/uniparser_result.json``.
    """

    DEFAULT_HOST = "https://uniparser.dp.tech/"
    TRIGGER_FILE_PATH = "trigger-file-async"
    TRIGGER_URL_PATH = "trigger-url-async"
    GET_FORMATTED_PATH = "get-formatted"

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        api_key: str = "",
        output_dir: str = "uniparser_result",
        parse_modes: Optional[Dict[str, str]] = None,
        output_flags: Optional[Dict[str, Any]] = None,
        format_flags: Optional[Dict[str, str]] = None,
        sync: bool = True,
        request_timeout: int = 1800,
        poll_max_retries: int = 120,
        poll_interval: int = 5,
    ) -> None:
        # 容错: 用户可能写 "https://uniparser.dp.tech" (无尾斜杠) 或加了 /api 后缀
        host = host.rstrip("/") + "/"
        # /api 是文档路径, 不是真正的 API 路径; trigger 接口直接挂在 host 下.
        if host.endswith("/api/"):
            host = host[: -len("api/")]
        self.host = host
        self.api_key = api_key
        self.output_dir = output_dir
        self.parse_modes = {**DEFAULT_PARSE_MODES, **(parse_modes or {})}
        self.output_flags = {**DEFAULT_OUTPUT_FLAGS, **(output_flags or {})}
        self.format_flags = {**DEFAULT_FORMAT_FLAGS, **(format_flags or {})}
        self.sync = sync
        self.request_timeout = request_timeout
        self.poll_max_retries = poll_max_retries
        self.poll_interval = poll_interval

    # ── 内部 HTTP helpers ────────────────────────────────────────────

    def _headers(self, json_body: bool = False) -> Dict[str, str]:
        h = {"X-API-Key": self.api_key}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _url(self, path: str) -> str:
        return self.host + path.lstrip("/")

    # ── 提交阶段 ─────────────────────────────────────────────────────

    def trigger_file(
        self, file_path: str, token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """上传单个 PDF 到 ``/trigger-file-async``。

        返回 ``{"token": str, "submit_response": dict}``. ``submit_response``
        是接口原始 JSON, 异步模式下含任务状态字段.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF 不存在或不是文件: {file_path}")

        token = token or _make_token(file_path)

        # form fields: token + sync + 解析配置 (各 SemanticType 一个数值字符串)
        data: Dict[str, str] = {
            "token": token,
            "sync": "true" if self.sync else "false",
        }
        for sem_type, mode in self.parse_modes.items():
            # 跳过禁用项 (=="0") 以减少 form 噪音; 接口也接受 "0"
            data[sem_type] = str(mode)

        url = self._url(self.TRIGGER_FILE_PATH)
        # 调用前打印实际下发参数, 方便用户排查 "改了 yaml 没生效?" 的疑问.
        # data 里除 token / sync 之外其它都是 SemanticType -> ParseMode, 全部记录.
        parse_payload = {k: v for k, v in data.items() if k not in ("token", "sync")}
        logger.info(
            f"[uniparser] >>> POST {url} token={token} sync={data['sync']} "
            f"parse_modes={parse_payload}"
        )
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    data=data,
                    files={"file": (os.path.basename(file_path), f)},
                    timeout=self.request_timeout,
                )
        except Exception as e:
            raise RuntimeError(f"UniParser trigger_file 请求失败: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"UniParser trigger_file HTTP {resp.status_code}: "
                f"{_safe_resp_text(resp)}"
            )

        try:
            body = resp.json()
        except Exception:
            body = {"raw": _safe_resp_text(resp, 1000)}

        logger.info(
            f"[uniparser] trigger_file ok: token={token} sync={self.sync} "
            f"http={resp.status_code}"
        )
        return {"token": token, "submit_response": body}

    # ── 结果获取阶段 ─────────────────────────────────────────────────

    def get_formatted(
        self,
        token: str,
        output_flags: Optional[Dict[str, Any]] = None,
        format_flags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """根据 ``token`` 拉取格式化结果。

        默认带上 ``DEFAULT_OUTPUT_FLAGS`` (content/objects/pages_dict/pages_tree
        全开) + ``DEFAULT_FORMAT_FLAGS`` (textual/table/equation/molecule 全部
        markdown). 调用方可通过参数覆写.
        """
        flags = {**self.output_flags, **(output_flags or {})}
        fmts = {**self.format_flags, **(format_flags or {})}

        payload: Dict[str, Any] = {"token": token}
        # 输出开关 (bool / 字符串都直接透传, 接口忽略未知字段)
        payload.update(flags)
        # 格式化标记 (每个 SemanticType 一个枚举字符串)
        payload.update(fmts)

        url = self._url(self.GET_FORMATTED_PATH)
        logger.info(
            f"[uniparser] >>> POST {url} token={token} output_flags={flags} format_flags={fmts}"
        )
        try:
            resp = requests.post(
                url,
                headers=self._headers(json_body=True),
                json=payload,
                timeout=self.request_timeout,
            )
        except Exception as e:
            raise RuntimeError(f"UniParser get_formatted 请求失败: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"UniParser get_formatted HTTP {resp.status_code}: "
                f"{_safe_resp_text(resp)}"
            )

        try:
            body = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"UniParser get_formatted 返回非 JSON: {_safe_resp_text(resp)}"
            ) from e

        # 调用后再回报: 实际返回里哪些字段拿到了, 各多大. 跟下发开关对照, 用户
        # 一眼能看出 "我开了 pages_tree 但产物没 pages_tree" 这种 mismatch.
        def _size(k: str) -> str:
            v = body.get(k)
            if v is None:
                return "None"
            if isinstance(v, str):
                return f"str(len={len(v)})"
            if isinstance(v, list):
                return f"list(len={len(v)})"
            return type(v).__name__
        present = ", ".join(
            f"{k}={_size(k)}" for k in
            ("content", "objects", "pages_dict", "pages_tree")
        )
        logger.info(f"[uniparser] <<< get_formatted http={resp.status_code} {present}")
        return body

    def poll_formatted(
        self, token: str, timeout_remaining: Optional[int] = None,
    ) -> Dict[str, Any]:
        """异步模式下轮询 ``/get-formatted`` 直到拿到非空结果。

        判定 "拿到非空结果" 的标准:
        - ``content`` 字段是非空字符串, 或
        - ``objects`` / ``pages_dict`` / ``pages_tree`` 任一字段是非空 list
        - 否则视为还在处理中, 等待 ``poll_interval`` 后重试.

        ``Status_Not_Found`` / ``Result_Not_Found`` 等错误会立即抛出, 不重试.
        """
        t0 = time.time()
        last_body: Dict[str, Any] = {}
        for attempt in range(self.poll_max_retries):
            if timeout_remaining and (time.time() - t0) >= timeout_remaining:
                raise TimeoutError(
                    f"UniParser 异步轮询超时 (剩余 {timeout_remaining}s 已耗尽)"
                )
            try:
                body = self.get_formatted(token)
                last_body = body
            except RuntimeError as e:
                msg = str(e)
                # 这些错误意味着任务/结果不存在, 没必要继续等
                fatal_kw = (
                    "Result_Not_Found",
                    "Status_Not_Found",
                    "Result_Decode_Failed",
                    "Status_Decode_Failed",
                )
                if any(kw in msg for kw in fatal_kw):
                    raise
                logger.info(
                    f"[uniparser][{attempt + 1}] get_formatted 暂态错误: "
                    f"{msg[:200]}"
                )
                time.sleep(self.poll_interval)
                continue

            if self._is_ready(body):
                logger.info(f"[uniparser] token={token} 解析就绪 (轮询 {attempt + 1} 次)")
                return body
            logger.info(
                f"[uniparser][{attempt + 1}] token={token} 暂无结果, "
                f"等待 {self.poll_interval}s..."
            )
            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"UniParser 异步轮询超出最大次数 {self.poll_max_retries}, "
            f"最后一次响应: {str(last_body)[:300]}"
        )

    @staticmethod
    def _is_ready(body: Dict[str, Any]) -> bool:
        if not isinstance(body, dict):
            return False
        content = body.get("content")
        if isinstance(content, str) and content.strip():
            return True
        for k in ("objects", "pages_dict", "pages_tree"):
            v = body.get(k)
            if isinstance(v, list) and len(v) > 0:
                return True
        return False

    # ── 完整流程 ─────────────────────────────────────────────────────

    def process(
        self,
        file_paths: List[str],
        timeout: Optional[int] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """完整流程: 逐个上传 → (异步)轮询 → 拉结果 → 落盘。

        Args:
            file_paths: PDF 文件路径列表
            timeout: 整个流程的超时秒数, 单文件耗时不会拆 (粗粒度统一计时);
                None 表示不限时 (但每次 HTTP 仍受 ``request_timeout`` 约束).
            save: 是否落盘. False 时只返回 result, 不写文件.

        Returns:
            {
              "output_dir": str,                # 顶级输出目录 (同 self.output_dir)
              "results": [                      # 每个 PDF 一项
                {
                  "file_path": str,
                  "pdf_stem": str,
                  "token": str,
                  "sub_dir": str,              # <output_dir>/<pdf_stem>/
                  "result_path": str | None,   # uniparser_result.json
                  "meta_path": str | None,     # uniparser_meta.json
                  "result": dict,              # /get-formatted 原始响应
                }, ...
              ],
              "tokens": List[str],
            }
        """
        if not file_paths:
            raise ValueError("file_paths 为空")
        if not self.api_key:
            raise RuntimeError("UniParser api_key 未配置, 无法调用 API")

        t_start = time.time()

        def _check_timeout() -> None:
            if timeout and (time.time() - t_start) > timeout:
                raise TimeoutError(f"UniParser 总耗时超过 {timeout}s")

        if save:
            os.makedirs(self.output_dir, exist_ok=True)

        results: List[Dict[str, Any]] = []
        tokens: List[str] = []
        for fp in file_paths:
            _check_timeout()
            logger.info(f"[uniparser] >>> {fp}")

            submit = self.trigger_file(fp)
            token = submit["token"]
            tokens.append(token)
            _check_timeout()

            # sync 模式下 trigger 已经等到完成, 直接 fetch; 异步模式要轮询
            remaining = None
            if timeout:
                remaining = max(1, int(timeout - (time.time() - t_start)))
            if self.sync:
                # sync=true 时, 直接 fetch 一次; 没就绪兜底进入 poll
                formatted = self.get_formatted(token)
                if not self._is_ready(formatted):
                    logger.info(
                        f"[uniparser] sync 返回空结果, 兜底改异步轮询 "
                        f"token={token}"
                    )
                    formatted = self.poll_formatted(
                        token, timeout_remaining=remaining,
                    )
            else:
                formatted = self.poll_formatted(
                    token, timeout_remaining=remaining,
                )

            pdf_stem = os.path.splitext(os.path.basename(fp))[0]
            sub_dir = os.path.join(self.output_dir, pdf_stem)
            result_path: Optional[str] = None
            meta_path: Optional[str] = None
            if save:
                os.makedirs(sub_dir, exist_ok=True)
                result_path = os.path.join(sub_dir, "uniparser_result.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(formatted, f, ensure_ascii=False, indent=2)
                meta_path = os.path.join(sub_dir, "uniparser_meta.json")
                # 关键: 把实际下发的 endpoint + 所有 flag 全部固化到 sidecar,
                # 用户可以直接 cat 这个文件确认 "我 yaml 改了 pages_tree=false,
                # 这次调用确实是用 false 跑的"; 也方便复现 / bug report 时贴出.
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "file_path": os.path.abspath(fp),
                            "pdf_stem": pdf_stem,
                            "token": token,
                            "endpoints": {
                                "trigger": self._url(self.TRIGGER_FILE_PATH),
                                "fetch": self._url(self.GET_FORMATTED_PATH),
                            },
                            "submit_response": submit["submit_response"],
                            "request_params": {
                                "sync": self.sync,
                                "parse_modes": self.parse_modes,
                                "output_flags": self.output_flags,
                                "format_flags": self.format_flags,
                            },
                            "result_payload_summary": {
                                "content_len": (
                                    len(formatted.get("content"))
                                    if isinstance(formatted.get("content"), str)
                                    else None
                                ),
                                "objects_len": (
                                    len(formatted.get("objects"))
                                    if isinstance(formatted.get("objects"), list)
                                    else None
                                ),
                                "pages_dict_len": (
                                    len(formatted.get("pages_dict"))
                                    if isinstance(formatted.get("pages_dict"), list)
                                    else None
                                ),
                                "pages_tree_len": (
                                    len(formatted.get("pages_tree"))
                                    if isinstance(formatted.get("pages_tree"), list)
                                    else None
                                ),
                            },
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                logger.info(f"[uniparser] 已保存: {result_path}")

            results.append(
                {
                    "file_path": fp,
                    "pdf_stem": pdf_stem,
                    "token": token,
                    "sub_dir": sub_dir,
                    "result_path": result_path,
                    "meta_path": meta_path,
                    "result": formatted,
                }
            )

        return {
            "output_dir": self.output_dir,
            "results": results,
            "tokens": tokens,
        }
