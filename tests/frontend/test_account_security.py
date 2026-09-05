"""Account layer: the security properties, asserted against the real files.

These check the properties that a code review would look for, and that a live
integration test could not prove anyway: ownership is enforced by the database
rather than the client, no secret is committed, and an unconfigured deployment
behaves exactly as it did before accounts existed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
ACCOUNT = ROOT / "frontend" / "src" / "lib" / "account.js"
SCHEMA = ROOT / "supabase" / "schema.sql"
node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


# ------------------------------------------------------- database enforcement
def test_row_level_security_is_enabled_on_the_table():
    sql = SCHEMA.read_text(encoding="utf-8")
    assert "enable row level security" in sql.lower()


def test_every_operation_is_restricted_to_the_authenticated_user():
    """A missing policy for any verb is a hole; check all four explicitly."""
    sql = SCHEMA.read_text(encoding="utf-8").lower()
    for verb in ("select", "insert", "update", "delete"):
        assert f"for {verb}" in sql, f"no policy governs {verb}"
    # and each one is pinned to the verified identity
    assert sql.count("auth.uid() = user_id") >= 4


def test_owner_defaults_to_the_verified_token_not_a_client_value():
    sql = SCHEMA.read_text(encoding="utf-8").lower()
    assert "default auth.uid()" in sql


def test_the_client_never_sends_a_user_id_when_writing():
    """A client-supplied owner is the classic broken-ownership bug."""
    src = ACCOUNT.read_text(encoding="utf-8")
    save = src[src.index("export async function saveEvent"):]
    save = save[:save.index("\n}")]
    assert "user_id" not in save, "saveEvent must not send user_id"


def test_account_deletion_can_only_delete_the_caller():
    sql = SCHEMA.read_text(encoding="utf-8").lower()
    fn = sql[sql.index("delete_own_account"):]
    assert "uid uuid := auth.uid()" in fn
    assert "raise exception 'not authenticated'" in fn
    # it must not accept an id argument
    assert re.search(r"delete_own_account\s*\(\s*\)", sql)


def test_the_definer_function_is_not_executable_anonymously():
    sql = SCHEMA.read_text(encoding="utf-8").lower()
    assert "revoke all on function public.delete_own_account() from public, anon" in sql
    assert "grant execute on function public.delete_own_account() to authenticated" in sql


# --------------------------------------------------------------- secrets
def test_no_supabase_credentials_are_committed():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    # Match a VALUE, not the word. The first version flagged SETUP.md for
    # warning against the service_role key, which is the opposite of a leak.
    #   - a real JWT: three base64 segments, and any Supabase key is one
    #   - an assignment of a service key to something that looks like a value
    jwt = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
    assigned = re.compile(
        r"(?i)(service_role|SUPABASE_SERVICE[A-Z_]*|ANON_KEY)\s*[:=]\s*[\"']?[A-Za-z0-9._-]{20,}")
    for rel in tracked:
        f = ROOT / rel
        if not f.is_file() or f.suffix in {".png", ".jpg", ".onnx", ".csv"}:
            continue
        if rel.endswith("test_account_security.py"):
            continue          # this file necessarily contains the patterns
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        assert not jwt.search(text), f"a JWT appears to be committed in {rel}"
        assert not assigned.search(text), f"a key value appears to be committed in {rel}"


def test_env_local_is_gitignored():
    ig = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env.local" in ig
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout
    assert ".env.local" not in tracked


def test_setup_warns_against_the_service_role_key():
    doc = (ROOT / "supabase" / "SETUP.md").read_text(encoding="utf-8")
    assert "service_role" in doc and "bypasses RLS" in doc


# ------------------------------------------------- unconfigured still works
@node
def test_unconfigured_deployment_reports_accounts_disabled():
    src = f'''
import {{ accountsEnabled, signedIn, currentUser }} from "{ACCOUNT.as_uri()}";
globalThis.localStorage = {{ getItem: () => null, setItem: () => {{}}, removeItem: () => {{}} }};
console.log(JSON.stringify({{ enabled: accountsEnabled(), signedIn: signedIn(),
                              user: currentUser() }}));
'''
    p = subprocess.run(["node", "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout.strip())
    assert got["enabled"] is False
    assert got["signedIn"] is False
    assert got["user"] is None


@node
def test_writes_are_refused_when_not_signed_in():
    src = f'''
import {{ saveEvent }} from "{ACCOUNT.as_uri()}";
globalThis.localStorage = {{ getItem: () => null, setItem: () => {{}}, removeItem: () => {{}} }};
try {{ await saveEvent({{ eventId: "x", timestamp: "2026-01-01" }}); console.log('"NO_ERROR"'); }}
catch (e) {{ console.log(JSON.stringify(e.message)); }}
'''
    p = subprocess.run(["node", "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "Not signed in" in json.loads(p.stdout.strip())


@node
def test_an_expired_session_is_treated_as_signed_out():
    src = f'''
import {{ signedIn }} from "{ACCOUNT.as_uri()}";
const past = Math.floor(Date.now() / 1000) - 60;
const store = {{ "aedt.session": JSON.stringify(
  {{ access_token: "t", expires_at: past, user: {{ id: "u", email: "a@b.c" }} }}) }};
globalThis.localStorage = {{ getItem: (k) => store[k] ?? null,
  setItem: () => {{}}, removeItem: () => {{}} }};
console.log(JSON.stringify(signedIn()));
'''
    p = subprocess.run(["node", "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.strip()) is False


def test_research_pipeline_never_reads_the_account_table():
    """Live check-ins must not leak into the scientific dataset."""
    hits = subprocess.run(["git", "grep", "-l", "twin_events", "--", "aedt/", "scripts/"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert not hits, f"research code references the account table: {hits}"
