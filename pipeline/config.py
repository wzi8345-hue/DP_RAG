"""集中式配置管理: 加载 YAML 配置, 支持多层覆盖 (默认 < 文件 < 环境变量 < 运行时)。"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# 尝试导入 yaml, 不可用时给出提示
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"

# 缓存单例
_cfg: Optional["Config"] = None


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并两个字典, override 覆盖 base。"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _resolve_env_vars(d: Dict) -> Dict:
    """递归解析字符串值中的 ${ENV_VAR} 引用。

    若环境变量不存在: 返回空串 "" 并打 warning, 避免把字面量 "${VAR}" 传给下游 API。
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _resolve_env_vars(v)
        elif isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_name = v[2:-1]
            if env_name in os.environ:
                result[k] = os.environ[env_name]
            else:
                logger.warning(
                    f"[config] 环境变量 {env_name} 未设置 (config key: {k}), 使用空串代替"
                )
                result[k] = ""
        else:
            result[k] = v
    return result


class Config:
    """Pipeline 配置, 按模块分节访问。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = _resolve_env_vars(data)

    # ── 分节访问 ──────────────────────────────────────────────────────

    @property
    def parsing(self) -> Dict[str, Any]:
        """PDF 解析后端选择 (backend: mineru | uniparser) 等通用解析参数。"""
        return self._data.setdefault("parsing", {})

    @property
    def mineru(self) -> Dict[str, Any]:
        return self._data.setdefault("mineru", {})

    @property
    def uniparser(self) -> Dict[str, Any]:
        return self._data.setdefault("uniparser", {})

    @property
    def chunking(self) -> Dict[str, Any]:
        return self._data.setdefault("chunking", {})

    @property
    def embedding(self) -> Dict[str, Any]:
        return self._data.setdefault("embedding", {})

    @property
    def milvus(self) -> Dict[str, Any]:
        return self._data.setdefault("milvus", {})

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._data.setdefault("retrieval", {})

    @property
    def generation(self) -> Dict[str, Any]:
        gen = self._data.setdefault("generation", {})
        # 若指定了 system_prompt_path, 从文件加载内容覆盖 system_prompt
        path = gen.get("system_prompt_path")
        if path and isinstance(path, str):
            p = Path(path)
            if not p.is_absolute():
                # 相对路径: 优先基于 CWD, 其次基于 config 文件所在目录
                p_cwd = Path.cwd() / p
                p_cfg = _DEFAULT_CONFIG_PATH.parent / p
                p = p_cwd if p_cwd.exists() else p_cfg
            if p.exists():
                gen["system_prompt"] = p.read_text(encoding="utf-8").strip()
            else:
                logger.warning(f"[config] system_prompt_path 不存在: {path}")
        return gen

    # ── 通用访问 ──────────────────────────────────────────────────────

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """支持点分路径访问, 如 config.get('mineru.poll.interval', 5)。"""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """支持点分路径设置。"""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        # 隐藏 api_key 等敏感字段
        safe = copy.deepcopy(self._data)
        for section in safe.values():
            if isinstance(section, dict):
                for k in list(section.keys()):
                    if "key" in k.lower() or "token" in k.lower() or "authorization" in k.lower():
                        section[k] = "***"
        return f"Config({safe})"


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Config:
    """加载配置: 默认配置 ← YAML 文件 ← 运行时覆盖。"""
    # 1) 加载默认配置
    if yaml is None:
        raise ImportError("需要 PyYAML: pip install pyyaml")
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    # 2) 加载用户配置 (如有)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        base = _deep_merge(base, user_cfg)

    # 3) 运行时覆盖
    if overrides:
        base = _deep_merge(base, overrides)

    return Config(base)


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置单例。"""
    global _cfg
    if _cfg is None:
        _cfg = load_config(config_path)
    return _cfg


def reset_config() -> None:
    """重置全局配置 (测试用)。"""
    global _cfg
    _cfg = None
