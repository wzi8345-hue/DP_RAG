"""LangGraph Agent 可观测性指标 (Prometheus)。

指标均为可选依赖, prometheus_client 未安装时静默跳过。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 延迟导入: prometheus_client 为可选依赖
_Counter = None
_Histogram = None

try:
    from prometheus_client import Counter, Histogram
    _Counter = Counter
    _Histogram = Histogram
except ImportError:
    pass


def _make_counter(name: str, doc: str, labels: Optional[list] = None):
    if _Counter is None:
        return _NoopMetric()
    try:
        return _Counter(name, doc, labels or [])
    except Exception:
        return _NoopMetric()


def _make_histogram(name: str, doc: str, labels: Optional[list] = None, buckets=None):
    if _Histogram is None:
        return _NoopMetric()
    try:
        kwargs = {}
        if labels:
            kwargs["labelnames"] = labels
        if buckets:
            kwargs["buckets"] = buckets
        return _Histogram(name, doc, **kwargs)
    except Exception:
        return _NoopMetric()


class _NoopMetric:
    """prometheus_client 不可用时的静默占位。"""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def time(self):
        return _NoopContext()


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# 核心指标
# ---------------------------------------------------------------------------

requests_total = _make_counter(
    "langgraph_requests_total",
    "Total LangGraph agent requests",
    labels=["status"],
)

request_duration_seconds = _make_histogram(
    "langgraph_request_duration_seconds",
    "End-to-end request latency",
    buckets=[1, 2, 5, 10, 15, 30, 60],
)

retry_total = _make_counter(
    "langgraph_retry_total",
    "Total retrieval retries triggered by self-reflection",
)

reflection_score = _make_histogram(
    "langgraph_reflection_score",
    "Reflection quality score distribution (0=skip, 1=insufficient, 2=sufficient)",
    buckets=[0, 1, 2],
)
