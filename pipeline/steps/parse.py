"""步骤 1: PDF 解析 — 支持 MinerU / UniParser 两条支路。

后端选择规则 (优先级从高到低):
1. ``kwargs["backend"]`` 显式传入
2. ``self.config.parsing.backend``
3. 兜底 ``"mineru"``

两个后端的输出 schema 不同:
- ``mineru``    -> ``mineru_result/<pdf_stem>/<pdf_stem>/*_content_list_v2.json``
- ``uniparser`` -> ``uniparser_result/<pdf_stem>/uniparser_result.json``

返回的 ``StepResult.data`` 里都带 ``backend`` 字段, 下游 (chunk 等) 据此分支.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseStep, StepResult, register_step
from ..clients.mineru import MinerUClient
from ..clients.uniparser import UniParserClient

logger = logging.getLogger(__name__)


def _resolve_backend(config: Any, kwargs: Dict[str, Any]) -> str:
    """从 kwargs / config 解析当前应该走哪条解析支路。"""
    backend = kwargs.get("backend")
    if not backend:
        parsing_cfg = getattr(config, "parsing", {}) or {}
        backend = parsing_cfg.get("backend") if isinstance(parsing_cfg, dict) else None
    backend = (backend or "mineru").strip().lower()
    if backend not in ("mineru", "uniparser"):
        raise ValueError(
            f"未知的 parsing.backend={backend!r}, 仅支持 mineru / uniparser"
        )
    return backend


def _run_mineru(config: Any, file_paths: List[str], output_dir: Optional[str],
                timeout: Optional[int]) -> Dict[str, Any]:
    cfg = config.mineru
    client = MinerUClient(
        api_url=cfg.get("api_url", MinerUClient.DEFAULT_API_URL),
        authorization=cfg.get("authorization", ""),
        model_version=cfg.get("model_version", "vlm"),
        output_dir=output_dir or cfg.get("output_dir", "mineru_result"),
        poll_max_retries=cfg.get("poll", {}).get("max_retries", 120),
        poll_interval=cfg.get("poll", {}).get("interval", 5),
    )
    return client.process(file_paths, timeout=timeout)


def _run_uniparser(config: Any, file_paths: List[str], output_dir: Optional[str],
                   timeout: Optional[int]) -> Dict[str, Any]:
    cfg = config.uniparser
    if not cfg.get("api_key"):
        raise RuntimeError(
            "uniparser.api_key 未配置 (建议 export UNIPARSER_API_KEY=...)"
        )
    client = UniParserClient(
        host=cfg.get("host", UniParserClient.DEFAULT_HOST),
        api_key=cfg.get("api_key", ""),
        output_dir=output_dir or cfg.get("output_dir", "uniparser_result"),
        parse_modes=cfg.get("parse_modes"),
        output_flags=cfg.get("output_flags"),
        format_flags=cfg.get("format_flags"),
        sync=bool(cfg.get("sync", True)),
        request_timeout=int(cfg.get("request_timeout", 1800)),
        poll_max_retries=int((cfg.get("poll") or {}).get("max_retries", 120)),
        poll_interval=int((cfg.get("poll") or {}).get("interval", 5)),
    )
    return client.process(file_paths, timeout=timeout)


@register_step
class ParseStep(BaseStep):
    """上传 PDF 到 MinerU 或 UniParser, 拿解析结果并落盘。"""

    name = "parse"

    def run(self, **kwargs) -> StepResult:
        file_paths: List[str] = kwargs.get("file_paths") or []
        if not file_paths:
            return StepResult(self.name, success=False, error="未提供 file_paths")

        output_dir = kwargs.get("output_dir")
        timeout = kwargs.get("timeout")

        try:
            backend = _resolve_backend(self.config, kwargs)
        except Exception as e:
            return StepResult(self.name, success=False, error=str(e))

        logger.info(f"[parse] backend={backend} files={file_paths}")

        try:
            if backend == "mineru":
                result = _run_mineru(self.config, file_paths, output_dir, timeout)
            else:
                result = _run_uniparser(self.config, file_paths, output_dir, timeout)
        except Exception as e:
            return StepResult(self.name, success=False, error=str(e))

        result["backend"] = backend
        return StepResult(self.name, success=True, data=result)
