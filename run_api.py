"""本机联调用 API 启动器：把根日志级别设为 INFO，便于查看 pipeline 完整日志。

用法（建议配合 run_api.sh）：
    CONFIG_PATH=local_api_config.yaml .venv-api/bin/python run_api.py
"""

import os

import uvicorn

from pipeline.logging_config import setup_logging

# 统一日志: 控制台 + 滚动文件 (logs/pipeline-api.log, 可用 LOG_FILE 覆盖)。
setup_logging(log_file=os.environ.get("LOG_FILE", "logs/pipeline-api.log"))

if __name__ == "__main__":
    uvicorn.run(
        "pipeline.api.app:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8080")),
        log_level="info",
    )
