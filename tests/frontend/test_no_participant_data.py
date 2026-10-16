"""The web application must not ship participant-level research records.

Mirrors frontend/scripts/check-no-participant-data.mjs so the rule also runs in
the Python test suite. Files are recognised by git blob SHA-1 only.
"""
import hashlib
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "frontend" / "src" / "data"
FORBIDDEN_BLOBS = {
    "74acce2ca5e4bc6782f38583cafced93a95fe3e2",  # participant summary statistics (removed)
    "6c464c45e4169a8fc04b4586c14a39e25dd10a03",  # participant report series and predictions (removed)
}
PEOPLE_LEVEL = {"participants.json", "trajectories.json"}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


DATA_FILES = sorted(DATA.glob("*.json"))


@pytest.mark.parametrize("path", DATA_FILES, ids=[p.name for p in DATA_FILES])
def test_no_removed_participant_file(path):
    raw = path.read_bytes()
    for data in (raw, raw.replace(b"\r\n", b"\n")):  # as stored, and as git stores it
        assert git_blob_sha1(data) not in FORBIDDEN_BLOBS, path.name


@pytest.mark.parametrize("path", [p for p in DATA_FILES if p.name in PEOPLE_LEVEL],
                         ids=[p.name for p in DATA_FILES if p.name in PEOPLE_LEVEL])
def test_people_level_files_are_marked_synthetic(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc.get("_synthetic") is True, path.name


def test_guard_hash_matches_git():
    """The hash used above is git's blob SHA-1 (checked on a known value)."""
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
