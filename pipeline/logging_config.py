"""统一日志配置: 一处定义级别 / 格式 / handler (控制台 + 滚动文件)。

替代散落在 ``run_api.py`` / ``pipeline/run.py`` / 各 eval 脚本里的多份
``logging.basicConfig``。提供 ``setup_logging()`` 作为唯一入口:

- 控制台 handler (stdout): 始终添加, 行为与原 basicConfig 一致。
- 滚动文件 handler: 仅当传入 ``log_file`` (或环境变量 ``LOG_FILE``) 时添加,
  用 ``RotatingFileHandler`` 限制单文件大小并保留若干份, 避免日志无限增长。

注意: ``RotatingFileHandler`` 非多进程安全。API 进程 (run_api.py) 写文件;
CLI (pipeline/run.py) 默认只输出控制台 (log_file=None), 避免与 API 抢同一文件
导致滚动竞争。
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

# 标记本模块加的 handler, 便于幂等判断与避免重复添加。
_HANDLER_TAG = "_dprag_handler"


def _level_from_env(default: int = logging.INFO) -> int:
    raw = (os.environ.get("LOG_LEVEL") or "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def setup_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[str] = None,
    *,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    force: bool = False,
) -> None:
    """配置 root logger (控制台 + 可选滚动文件), 幂等。

    Args:
        level: 日志级别 (int 或 "INFO"/"DEBUG" 等); None 时取 ``LOG_LEVEL`` 环境变量,
            再缺省回退 INFO。
        log_file: 滚动文件路径; None 表示只输出控制台。
        max_bytes: 单文件上限字节数; None 时取 ``LOG_MAX_BYTES`` (默认 50MB)。
        backup_count: 保留的历史文件份数; None 时取 ``LOG_BACKUP_COUNT`` (默认 5)。
        force: True 时移除本模块此前添加的 handler 后重新配置。
    """
    if isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.INFO)
    if level is None:
        level = _level_from_env()

    root = logging.getLogger()
    root.setLevel(level)

    existing = [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]
    if existing and not force:
        # 已配置过: 仅同步级别, 不重复加 handler。
        for h in existing:
            h.setLevel(level)
        # 确保 pipeline.* 的 INFO 真正产生 (root 默认 WARNING 会丢弃)
        logging.getLogger("pipeline").setLevel(min(level, logging.INFO))
        return
    for h in existing:
        root.removeHandler(h)

    fmt = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    setattr(console, _HANDLER_TAG, True)
    root.addHandler(console)

    log_file = log_file or os.environ.get("LOG_FILE")
    if log_file:
        if max_bytes is None:
            max_bytes = int(os.environ.get("LOG_MAX_BYTES", str(50 * 1024 * 1024)))
        if backup_count is None:
            backup_count = int(os.environ.get("LOG_BACKUP_COUNT", "5"))
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)

    # pipeline.* 的 INFO 日志必须真正产生, 否则 SessionLogHandler / 文件都收不到。
    logging.getLogger("pipeline").setLevel(min(level, logging.INFO))
