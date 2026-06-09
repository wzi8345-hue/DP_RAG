"""Logto JWT 本地校验。

前端用 ``getAccessToken(API_RESOURCE)`` 取到 Logto 颁发的 JWT access_token,
后端用 JWKS 公钥本地校验签名 + iss/aud/exp + scope, 无需每请求回 Logto。

环境变量:
- LOGTO_ISSUER          默认 https://auth.dplink.cc/oidc
- LOGTO_JWKS_URI        默认 https://auth.dplink.cc/oidc/jwks
- LOGTO_AUDIENCE        默认 https://funmg.dp.tech/sci-loop-api (API Resource)
- LOGTO_REQUIRED_SCOPE  默认 all:data (留空则不校验 scope)
- AUTH_DISABLED         1/true 时跳过校验, 用固定 dev 用户 (仅本地)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """单次请求的用户上下文。"""

    user_id: str                      # Logto sub
    org_id: str | None = None      # 组织 id (org-public 共享用); 可能为 None
    scopes: list[str] = field(default_factory=list)
    client_id: str = ""
    claims: dict[str, Any] = field(default_factory=dict)
    token: str = ""

    @property
    def is_dev(self) -> bool:
        return self.user_id == "dev"


_DEV_CONTEXT = AuthContext(
    user_id="dev", org_id="dev-org", scopes=["all:data"], client_id="dev",
)


def auth_disabled() -> bool:
    return os.environ.get("AUTH_DISABLED", "0").strip().lower() in ("1", "true", "yes")


class LogtoVerifier:
    """惰性持有 PyJWKClient, 校验并解析 JWT。"""

    def __init__(
        self,
        issuer: str,
        jwks_uri: str,
        audience: str,
        required_scope: str,
    ) -> None:
        self.issuer = issuer
        self.jwks_uri = jwks_uri
        self.audience = audience
        self.required_scope = required_scope
        self._jwk_client = None  # lazy: 避免无网络时 import 即失败

    def _client(self):
        if self._jwk_client is None:
            from jwt import PyJWKClient

            # PyJWKClient 自带 key 缓存 + 按需刷新
            self._jwk_client = PyJWKClient(self.jwks_uri, cache_keys=True, lifespan=3600)
        return self._jwk_client

    def verify(self, token: str) -> AuthContext:
        import jwt

        try:
            signing_key = self._client().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES384", "ES256", "RS256"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat"]},
            )
        except HTTPException:
            raise
        except Exception as e:  # 签名/过期/aud/iss 等
            logger.info("[auth] JWT 校验失败: %s", e)
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e

        scope_raw = claims.get("scope") or ""
        scopes = scope_raw.split() if isinstance(scope_raw, str) else list(scope_raw)
        if self.required_scope and self.required_scope not in scopes:
            raise HTTPException(status_code=403, detail="Insufficient scope")

        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Token missing sub")

        return AuthContext(
            user_id=sub,
            org_id=claims.get("organization_id") or claims.get("org_id"),
            scopes=scopes,
            client_id=claims.get("client_id", "") or claims.get("aud", "") if isinstance(claims.get("aud"), str) else claims.get("client_id", ""),
            claims=claims,
            token=token,
        )


_verifier: LogtoVerifier | None = None


def get_verifier() -> LogtoVerifier:
    global _verifier
    if _verifier is None:
        _verifier = LogtoVerifier(
            issuer=os.environ.get("LOGTO_ISSUER", "https://auth.dplink.cc/oidc"),
            jwks_uri=os.environ.get("LOGTO_JWKS_URI", "https://auth.dplink.cc/oidc/jwks"),
            audience=os.environ.get("LOGTO_AUDIENCE", "https://funmg.dp.tech/sci-loop-api"),
            required_scope=os.environ.get("LOGTO_REQUIRED_SCOPE", "all:data"),
        )
    return _verifier


def _extract_bearer(authorization: str) -> str:
    token = authorization.removeprefix("Bearer ").removeprefix("bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return token


async def require_auth(authorization: str = Header(default="")) -> AuthContext:
    """强制鉴权依赖。AUTH_DISABLED 时返回固定 dev 用户。"""
    if auth_disabled():
        return _DEV_CONTEXT
    return get_verifier().verify(_extract_bearer(authorization))


async def optional_auth(authorization: str = Header(default="")) -> AuthContext | None:
    """可选鉴权: 无 token 返回 None (AUTH_DISABLED 时返回 dev)。"""
    if not authorization:
        return _DEV_CONTEXT if auth_disabled() else None
    try:
        return await require_auth(authorization)
    except HTTPException:
        return None
