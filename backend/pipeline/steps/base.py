"""Pipeline 步骤基类与注册机制。"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StepResult:
    """步骤执行结果。"""

    def __init__(self, step_name: str, success: bool, data: Any = None,
                 elapsed: float = 0.0, error: Optional[str] = None) -> None:
        self.step_name = step_name
        self.success = success
        self.data = data
        self.elapsed = elapsed
        self.error = error

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error})"
        return f"StepResult({self.step_name}, {status}, {self.elapsed:.2f}s)"


class BaseStep(ABC):
    """所有 pipeline 步骤的基类。"""

    name: str = "base"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(f"pipeline.step.{self.name}")

    @abstractmethod
    def run(self, **kwargs) -> StepResult:
        """执行步骤, 返回 StepResult。"""

    def _execute(self, **kwargs) -> StepResult:
        """带计时和异常处理的执行包装。"""
        t0 = time.time()
        try:
            result = self.run(**kwargs)
            result.elapsed = time.time() - t0
            self.logger.info(f"{self.name} completed in {result.elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - t0
            self.logger.error(f"{self.name} failed: {e}")
            return StepResult(self.name, success=False, elapsed=elapsed, error=str(e))


# 步骤注册表
_STEP_REGISTRY: Dict[str, type] = {}


def register_step(cls: type) -> type:
    """注册步骤类。"""
    _STEP_REGISTRY[cls.name] = cls
    return cls


def get_step(name: str) -> type:
    """获取已注册的步骤类。"""
    if name not in _STEP_REGISTRY:
        raise KeyError(f"未注册的步骤: {name}, 可用: {list(_STEP_REGISTRY.keys())}")
    return _STEP_REGISTRY[name]


def list_steps() -> list:
    """列出所有已注册步骤。"""
    return list(_STEP_REGISTRY.keys())
