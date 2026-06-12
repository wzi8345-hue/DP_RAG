"""Shared authorization helpers for owner/org/public resources."""

from __future__ import annotations

from typing import Protocol

from fastapi import HTTPException

from ..auth import AuthContext
from ..db.models import Visibility


class OwnedResource(Protocol):
    owner_id: str
    org_id: str | None
    visibility: Visibility


class ScopedResource(Protocol):
    owner_id: str
    org_id: str | None


def can_read(auth: AuthContext, resource: OwnedResource) -> bool:
    if resource.owner_id == auth.user_id:
        return True
    if resource.visibility == "public":
        return True
    return (
        resource.visibility == "org"
        and bool(resource.org_id)
        and bool(auth.org_id)
        and resource.org_id == auth.org_id
    )


def can_write(auth: AuthContext, resource: OwnedResource) -> bool:
    return resource.owner_id == auth.user_id


def same_org(auth: AuthContext, resource: ScopedResource) -> bool:
    return bool(auth.org_id) and bool(resource.org_id) and auth.org_id == resource.org_id


def can_admin_read(auth: AuthContext, resource: ScopedResource) -> bool:
    if resource.owner_id == auth.user_id:
        return True
    if auth.is_root:
        return True
    return auth.role == "admin" and same_org(auth, resource)


def can_manage(auth: AuthContext, resource: ScopedResource) -> bool:
    return can_admin_read(auth, resource)


def can_delete(auth: AuthContext, resource: ScopedResource) -> bool:
    return can_manage(auth, resource)


def require_read(auth: AuthContext, resource: OwnedResource) -> None:
    if not can_read(auth, resource):
        raise HTTPException(status_code=403, detail="No read permission")


def require_write(auth: AuthContext, resource: OwnedResource) -> None:
    if not can_write(auth, resource):
        raise HTTPException(status_code=403, detail="No write permission")


def require_admin_read(auth: AuthContext, resource: ScopedResource) -> None:
    if not can_admin_read(auth, resource):
        raise HTTPException(status_code=403, detail="No admin read permission")


def require_manage(auth: AuthContext, resource: ScopedResource) -> None:
    if not can_manage(auth, resource):
        raise HTTPException(status_code=403, detail="No management permission")


def require_delete(auth: AuthContext, resource: ScopedResource) -> None:
    if not can_delete(auth, resource):
        raise HTTPException(status_code=403, detail="No delete permission")


def require_visibility_allowed(auth: AuthContext, visibility: Visibility) -> None:
    if visibility == "org" and not auth.org_id and not auth.is_root:
        raise HTTPException(status_code=400, detail="User does not belong to an organization")


def visibility_filter_sql(alias: str = "") -> str:
    """Return SQL predicate for private(mine) ∪ org ∪ public."""
    prefix = f"{alias}." if alias else ""
    return (
        f"({prefix}owner_id = %(user_id)s "
        f"OR {prefix}visibility = 'public' "
        f"OR ({prefix}visibility = 'org' AND {prefix}org_id IS NOT NULL AND {prefix}org_id = %(org_id)s))"
    )
