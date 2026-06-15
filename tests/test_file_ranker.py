"""Tests for controller/file_ranker.py — Phase 8 Knob P8-C pre-selection."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from code_capsules.controller.file_ranker import (
    _extract_signals,
    _module_path_candidates,
    rank_relevant_files,
    format_for_prompt,
)


# ── Signal extraction ────────────────────────────────────────────────────────

def test_extract_signals_finds_quoted_paths():
    text = "The bug is in 'src/foo/bar.py' and also in `lib/baz.py`."
    paths, _, _ = _extract_signals(text)
    assert "src/foo/bar.py" in paths
    assert "lib/baz.py" in paths


def test_extract_signals_finds_dotted_python_paths():
    text = "ModelField.contribute_to_class in django.db.models.fields breaks."
    _, dotted, _ = _extract_signals(text)
    assert any("django.db.models.fields" in d for d in dotted)


def test_extract_signals_extracts_identifiers():
    text = "When you call MyClass.do_thing() it raises ValueError on snake_case_input."
    _, _, idents = _extract_signals(text)
    assert "MyClass" in idents
    assert "do_thing" in idents
    assert "ValueError" in idents
    assert "snake_case_input" in idents
    # Stopwords are filtered
    assert "the" not in [i.lower() for i in idents]


def test_extract_signals_drops_short_dotted_components():
    text = "django.db.models — investigating db, fields"
    _, dotted, idents = _extract_signals(text)
    # db is in a dotted path and is short — should not appear as standalone ident
    assert "db" not in idents


# ── Module path expansion ────────────────────────────────────────────────────

def test_module_path_candidates_emits_multiple_depths():
    cands = _module_path_candidates("django.db.models.fields")
    assert "django/db/models/fields.py" in cands
    assert "django/db/models/fields/__init__.py" in cands
    assert "django/db/models.py" in cands or "django/db/models/__init__.py" in cands


# ── Ranking integration with a synthetic mini-repo ───────────────────────────

@pytest.fixture
def mini_repo(tmp_path: Path):
    """Create a small git repo with a handful of files."""
    repo = tmp_path / "minirepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)

    files = {
        "src/auth/login.py": (
            "class LoginView:\n"
            "    '''Handles user login.'''\n"
            "    def post(self, request):\n"
            "        return authenticate(request)\n"
        ),
        "src/auth/__init__.py": "",
        "src/utils/helpers.py": (
            "def authenticate(req):\n"
            "    return True\n"
        ),
        "src/models.py": (
            "class User:\n"
            "    pass\n"
        ),
        "tests/test_login.py": (
            "def test_login_view():\n"
            "    pass\n"
        ),
        "README.md": "# Mini repo",
    }
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_rank_quoted_path_wins(mini_repo: Path):
    issue = "The bug is in 'src/auth/login.py'. The LoginView.post handler fails."
    ranked = rank_relevant_files(mini_repo, issue, top_n=5)
    paths = [r.path for r in ranked]
    assert paths[0] == "src/auth/login.py"


def test_rank_identifier_only_match(mini_repo: Path):
    # No explicit path; only the identifier LoginView is mentioned
    issue = "Calling LoginView.post() blows up with an unexpected error."
    ranked = rank_relevant_files(mini_repo, issue, top_n=5)
    # login.py defines LoginView in its peek — should rank above unrelated files
    paths = [r.path for r in ranked]
    assert paths[0] == "src/auth/login.py", f"Expected login.py first, got: {paths}"


def test_rank_module_path_finds_corresponding_file(mini_repo: Path):
    issue = "src.utils.helpers.authenticate is mis-spelled."
    ranked = rank_relevant_files(mini_repo, issue, top_n=5)
    paths = [r.path for r in ranked]
    assert "src/utils/helpers.py" in paths


def test_rank_returns_at_most_top_n(mini_repo: Path):
    issue = "LoginView authenticate User test_login helpers"
    ranked = rank_relevant_files(mini_repo, issue, top_n=3)
    assert len(ranked) <= 3


def test_rank_returns_empty_for_unrelated_issue(mini_repo: Path):
    issue = "Unrelated jargon: foobarbazquxquux unknownsymbol."
    ranked = rank_relevant_files(mini_repo, issue, top_n=5)
    # No matches expected
    assert ranked == [] or all(r.score < 1.0 for r in ranked)


def test_rank_filters_by_extension(mini_repo: Path):
    issue = "Look at the README.md and src/auth/login.py"
    ranked = rank_relevant_files(mini_repo, issue, top_n=10, extensions=("py",))
    paths = [r.path for r in ranked]
    assert "README.md" not in paths


# ── Prompt formatting ────────────────────────────────────────────────────────

def test_format_for_prompt_empty_returns_empty_string():
    assert format_for_prompt([]) == ""


def test_format_for_prompt_lists_paths_in_order():
    from code_capsules.controller.file_ranker import RankedFile
    ranked = [
        RankedFile(path="a/b.py", score=5.0, matched_tokens=()),
        RankedFile(path="c/d.py", score=3.0, matched_tokens=()),
    ]
    out = format_for_prompt(ranked)
    assert "## Likely relevant files" in out
    assert "`a/b.py`" in out
    assert "`c/d.py`" in out
    # Order preserved
    assert out.index("a/b.py") < out.index("c/d.py")
