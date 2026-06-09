"""Per-(route, stage, chunk_type) 质量阈值矩阵 (P1.1, P1.2 #6 修复)。

旧版本只有 `quality_threshold_by_type`, 全局生效, 且诊断层完全无视, 用 `default_threshold`
判 alive/dead/too_narrow/too_broad. 这一层把两边统一到同一份 RouteThresholds 实例:

  - reranker_node._route_threshold_for() : per-route gate
  - rerank_diagnosis.R3/R4/R5            : alive/dead/narrow/broad 判定

查找顺序 (`for_(route, stage, chunk_type)`):
  1. by_route[route][stage][type]
  2. by_route[route][stage]["default"]
  3. by_route[route][type]                      (route 无 stage 分层)
  4. by_route[route]["default"]
  5. by_type[type]
  6. default

YAML schema 示例:

    quality_thresholds:
      default: 0.25
      by_type:                         # 全局 type 兜底, 兼容旧 quality_threshold_by_type
        text: 0.30
        image: 0.10
        table: 0.10
        references: 0.15
      by_route:
        summary:
          default: 0.35                # summary 所有 type 都用此阈值
        progressive:
          l1:                          # progressive level 1 (doc anchor) — 宽松
            text: 0.18
            default: 0.12
          l2:                          # doc-scoped chunk drill — 严格
            text: 0.32
            image: 0.15
            table: 0.15
            references: 0.20
            default: 0.18
          l2_global:                   # 全库 fallback — 最严格
            text: 0.35
            default: 0.20
          default: 0.25                # 没打 stage 的兜底
        local:
          text: 0.30
          image: 0.12
          table: 0.12
          references: 0.20
          default: 0.18
        metadata:                      # 仅 topk_only 命中走到这里
          text: 0.22
          default: 0.12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

# 哨兵: 表示某层未配置, 由调用方继续向下找
_MISSING = object()


@dataclass(frozen=True)
class RouteThresholds:
    """阶梯式查找的 per-(route, stage, chunk_type) 阈值矩阵."""
    default: float = 0.25
    by_type: Mapping[str, float] = field(default_factory=dict)
    by_route: Mapping[str, Any] = field(default_factory=dict)

    def for_(
        self,
        route: Optional[str],
        stage: Optional[str],
        chunk_type: Optional[str],
    ) -> Tuple[float, str]:
        """返回 (threshold, source_path); source_path 用于日志/调试."""
        ct = (chunk_type or "").strip().lower()
        rt = (route or "").strip().lower()
        st = (stage or "").strip().lower()

        # 1. by_route[route][stage][type]  ②. by_route[route][stage].default
        if rt and rt in self.by_route:
            route_cfg = self.by_route[rt]
            if isinstance(route_cfg, Mapping):
                if st and st in route_cfg:
                    stage_cfg = route_cfg[st]
                    if isinstance(stage_cfg, Mapping):
                        if ct and ct in stage_cfg:
                            v = _coerce_float(stage_cfg[ct])
                            if v is not None:
                                return v, f"by_route[{rt}][{st}][{ct}]"
                        if "default" in stage_cfg:
                            v = _coerce_float(stage_cfg["default"])
                            if v is not None:
                                return v, f"by_route[{rt}][{st}].default"
                    else:
                        # by_route[rt][st] 直接是 float (e.g., 简写形式)
                        v = _coerce_float(stage_cfg)
                        if v is not None:
                            return v, f"by_route[{rt}][{st}]"
                # 3. by_route[route][type]  ④. by_route[route].default
                if ct and ct in route_cfg:
                    val = route_cfg[ct]
                    if not isinstance(val, Mapping):
                        v = _coerce_float(val)
                        if v is not None:
                            return v, f"by_route[{rt}][{ct}]"
                if "default" in route_cfg:
                    val = route_cfg["default"]
                    if not isinstance(val, Mapping):
                        v = _coerce_float(val)
                        if v is not None:
                            return v, f"by_route[{rt}].default"

        # 5. by_type[type]
        if ct and ct in self.by_type:
            v = _coerce_float(self.by_type[ct])
            if v is not None:
                return v, f"by_type[{ct}]"

        # 6. default
        return float(self.default), "default"

    def for_chunk_type(self, chunk_type: Optional[str]) -> float:
        """诊断层 R3/R4/R5 没有 route/stage 上下文时的便捷接口。"""
        return self.for_(None, None, chunk_type)[0]

    @classmethod
    def from_dict(
        cls,
        raw: Optional[Dict[str, Any]],
        *,
        legacy_default: Optional[float] = None,
        legacy_by_type: Optional[Mapping[str, float]] = None,
    ) -> "RouteThresholds":
        """从 YAML dict 构造, 同时吸收旧的 `quality_threshold` / `quality_threshold_by_type`。

        Args:
            raw: `quality_thresholds` 子表
            legacy_default: 旧 `quality_threshold` 全局值 (e.g., 0.25), 没新表时兜底
            legacy_by_type: 旧 `quality_threshold_by_type` (e.g., {text:0.30,...})
        """
        raw = raw or {}
        # default: 新 schema 优先; 否则用旧值; 否则 0.25
        if "default" in raw:
            default_v = _coerce_float(raw["default"], 0.25) or 0.25
        elif legacy_default is not None:
            default_v = float(legacy_default)
        else:
            default_v = 0.25

        # by_type: 新 schema 优先; 否则用旧的 quality_threshold_by_type
        by_type_raw = raw.get("by_type")
        if isinstance(by_type_raw, Mapping):
            by_type = {str(k).lower(): float(v) for k, v in by_type_raw.items() if v is not None}
        elif legacy_by_type:
            by_type = {str(k).lower(): float(v) for k, v in legacy_by_type.items() if v is not None}
        else:
            by_type = {}

        # by_route: 深拷贝 + lowercase keys
        by_route_raw = raw.get("by_route") or {}
        by_route: Dict[str, Any] = {}
        if isinstance(by_route_raw, Mapping):
            for route_key, route_cfg in by_route_raw.items():
                rk = str(route_key).lower()
                by_route[rk] = _normalize_nested(route_cfg)

        return cls(default=default_v, by_type=by_type, by_route=by_route)


def _coerce_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _normalize_nested(node: Any) -> Any:
    """递归 lowercase mapping keys, 数值转 float."""
    if isinstance(node, Mapping):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            kl = str(k).lower()
            out[kl] = _normalize_nested(v)
        return out
    coerced = _coerce_float(node)
    return coerced if coerced is not None else node
