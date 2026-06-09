"""Logto 鉴权: 本地校验 JWT access_token + 用户/组织上下文。

对外暴露:
- AuthContext: 当前请求的用户上下文 (user_id / org_id / scopes ...)
- require_auth: FastAPI 依赖, 强制鉴权, 返回 AuthContext
- optional_auth: 可选鉴权 (运维只读接口可用), 失败返回 None
"""

from .logto import AuthContext, optional_auth, require_auth

__all__ = ["AuthContext", "require_auth", "optional_auth"]
