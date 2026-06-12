from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("pipeline")
auth_pkg = _ensure_package("pipeline.auth")
_ensure_package("pipeline.api")
db_pkg = _ensure_package("pipeline.db")

logto = _load_module("pipeline.auth.logto", ROOT / "auth" / "logto.py")
auth_pkg.AuthContext = logto.AuthContext
models = _load_module("pipeline.db.models", ROOT / "db" / "models.py")
db_pkg.models = models
authz = _load_module("pipeline.api.authz", ROOT / "api" / "authz.py")

AuthContext = logto.AuthContext
Visibility = models.Visibility
can_admin_read = authz.can_admin_read
can_read = authz.can_read
require_visibility_allowed = authz.require_visibility_allowed
_auth_role_from_claims = logto._auth_role_from_claims


@dataclass
class Resource:
    owner_id: str
    org_id: str | None
    visibility: Visibility = "private"


class TestAuthzRoles(unittest.TestCase):
    def test_parses_highest_org_role_and_org(self):
        role, org_id, organizations, organization_roles = _auth_role_from_claims(
            {
                "organizations": ["org-user", "org-admin"],
                "organization_roles": [
                    "org-user:sci-loop-user",
                    "org-admin:sci-loop-admin",
                ],
            }
        )

        self.assertEqual(role, "admin")
        self.assertEqual(org_id, "org-admin")
        self.assertEqual(organizations, ["org-user", "org-admin"])
        self.assertEqual(organization_roles[-1], "org-admin:sci-loop-admin")

    def test_root_role_wins(self):
        role, org_id, _, _ = _auth_role_from_claims(
            {
                "organization_roles": [
                    "org-a:sci-loop-admin",
                    "org-root:sci-loop-root",
                ],
            }
        )

        self.assertEqual(role, "root")
        self.assertEqual(org_id, "org-root")

    def test_normal_read_excludes_private_same_org(self):
        auth = AuthContext(user_id="u2", org_id="org-a", role="user")
        resource = Resource(owner_id="u1", org_id="org-a", visibility="private")

        self.assertFalse(can_read(auth, resource))
        self.assertFalse(can_admin_read(auth, resource))

    def test_org_admin_can_admin_read_same_org_private(self):
        auth = AuthContext(user_id="admin", org_id="org-a", role="admin")
        resource = Resource(owner_id="u1", org_id="org-a", visibility="private")

        self.assertTrue(can_admin_read(auth, resource))

    def test_org_admin_cannot_admin_read_other_org_private(self):
        auth = AuthContext(user_id="admin", org_id="org-a", role="admin")
        resource = Resource(owner_id="u1", org_id="org-b", visibility="private")

        self.assertFalse(can_admin_read(auth, resource))

    def test_root_can_admin_read_without_org(self):
        auth = AuthContext(user_id="root", role="root")
        resource = Resource(owner_id="u1", org_id=None, visibility="private")

        self.assertTrue(can_admin_read(auth, resource))

    def test_user_without_org_cannot_set_org_visibility(self):
        with self.assertRaises(HTTPException):
            require_visibility_allowed(AuthContext(user_id="u1"), "org")


if __name__ == "__main__":
    unittest.main()
