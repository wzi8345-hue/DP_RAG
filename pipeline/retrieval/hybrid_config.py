"""Hybrid RRF 权重配置 (#14): router retrieve_bias → 分阶段权重映射。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


VALID_RETRIEVE_BIASES = frozenset({"semantic", "entity_heavy", "keyword", "balanced"})


@dataclass(frozen=True)
class WeightPair:
    dense: float
    bm25: float


@dataclass(frozen=True)
class HybridWeightConfig:
    """retrieve_bias 枚举 → 各 stage 的 dense/bm25 权重。"""

    mode: str = "router"  # router | static
    static_dense: float = 0.7
    static_bm25: float = 0.3
    bias_profiles: Dict[str, Dict[str, WeightPair]] = field(default_factory=dict)
    clamp_min: float = 0.25
    clamp_max: float = 0.85

    def __post_init__(self) -> None:
        if self.mode not in ("router", "static"):
            object.__setattr__(self, "mode", "router")


_DEFAULT_STAGE_PROFILES: Dict[str, WeightPair] = {
    "progressive_l1": WeightPair(0.65, 0.35),
    "progressive_l1_global": WeightPair(0.50, 0.50),
    "progressive_l2": WeightPair(0.55, 0.45),
    "local_l2": WeightPair(0.45, 0.55),
    "simple": WeightPair(0.60, 0.40),
}

_DEFAULT_BIAS_PROFILES: Dict[str, Dict[str, WeightPair]] = {
    "balanced": dict(_DEFAULT_STAGE_PROFILES),
    "semantic": {
        "progressive_l1": WeightPair(0.75, 0.25),
        "progressive_l1_global": WeightPair(0.60, 0.40),
        "progressive_l2": WeightPair(0.65, 0.35),
        "local_l2": WeightPair(0.55, 0.45),
        "simple": WeightPair(0.70, 0.30),
    },
    "entity_heavy": {
        "progressive_l1": WeightPair(0.50, 0.50),
        "progressive_l1_global": WeightPair(0.35, 0.65),
        "progressive_l2": WeightPair(0.40, 0.60),
        "local_l2": WeightPair(0.30, 0.70),
        "simple": WeightPair(0.45, 0.55),
    },
    "keyword": {
        "progressive_l1": WeightPair(0.55, 0.45),
        "progressive_l1_global": WeightPair(0.40, 0.60),
        "progressive_l2": WeightPair(0.45, 0.55),
        "local_l2": WeightPair(0.35, 0.65),
        "simple": WeightPair(0.50, 0.50),
    },
}


DEFAULT_HYBRID_CONFIG = HybridWeightConfig(
    bias_profiles={k: dict(v) for k, v in _DEFAULT_BIAS_PROFILES.items()},
)


def _parse_pair(raw: Any, default: WeightPair) -> WeightPair:
    if not isinstance(raw, dict):
        return default
    return WeightPair(
        dense=float(raw.get("dense", default.dense)),
        bm25=float(raw.get("bm25", default.bm25)),
    )


def hybrid_config_from_dict(raw: Optional[Dict[str, Any]]) -> HybridWeightConfig:
    raw = raw or {}
    bias_profiles = {k: dict(v) for k, v in _DEFAULT_BIAS_PROFILES.items()}

    raw_profiles = raw.get("bias_profiles") or {}
    if isinstance(raw_profiles, dict):
        for bias_name, stages in raw_profiles.items():
            if bias_name not in VALID_RETRIEVE_BIASES:
                continue
            if not isinstance(stages, dict):
                continue
            merged = dict(bias_profiles.get(bias_name, _DEFAULT_STAGE_PROFILES))
            for stage, default in _DEFAULT_STAGE_PROFILES.items():
                if stage in stages:
                    merged[stage] = _parse_pair(stages[stage], default)
            bias_profiles[bias_name] = merged

    clamp_raw = raw.get("clamp") or [0.25, 0.85]
    clamp_min = float(clamp_raw[0]) if len(clamp_raw) > 0 else 0.25
    clamp_max = float(clamp_raw[1]) if len(clamp_raw) > 1 else 0.85

    mode = str(raw.get("mode", "router"))
    if mode == "adaptive":
        mode = "router"

    return HybridWeightConfig(
        mode=mode,
        static_dense=float(raw.get("static_dense_weight", raw.get("static_dense", 0.7))),
        static_bm25=float(raw.get("static_bm25_weight", raw.get("static_bm25", 0.3))),
        bias_profiles=bias_profiles,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
