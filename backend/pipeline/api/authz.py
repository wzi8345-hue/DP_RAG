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


def require_read(auth: AuthContext, resource: OwnedResource) -> None:
    if not can_read(auth, resource):
        raise HTTPException(status_code=403, detail="No read permission")


def require_write(auth: AuthContext, resource: OwnedResource) -> None:
    if not can_write(auth, resource):
        raise HTTPException(status_code=403, detail="No write permission")


def visibility_filter_sql(alias: str = "") -> str:
    """Return SQL predicate for private(mine) ∪ org ∪ public."""
    prefix = f"{alias}." if alias else ""
    return (
        f"({prefix}owner_id = %(user_id)s "
        f"OR {prefix}visibility = 'public' "
        f"OR ({prefix}visibility = 'org' AND {prefix}org_id IS NOT NULL AND {prefix}org_id = %(org_id)s))"
    )
