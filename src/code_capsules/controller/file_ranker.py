"""
Phase 8 Knob P8-C: Issue-relevant file pre-selection.

Front-loads context by ranking repo files against the SWE-bench issue text and
returning the top-N most likely-relevant paths. The harness then prepends a
"## Likely relevant files" block to the prompt, cutting the model's exploration
overhead.

Mechanism (Okapi BM25 over file content + path, no external dependencies):
  1. Extract candidate signals from the issue text (the BM25 query):
       - dotted Python paths (e.g. `django.db.models.fields`)
       - quoted file paths (e.g. ``'path/to/file.py'``)
       - identifiers (CamelCase, snake_case, ALL_CAPS) of length >= 3
       - traceback file mentions
  2. Treat each tracked repo file as a document: the tokens of its path plus the
     tokens of its first-2 KB peek. Score the query against every document with
     Okapi BM25 (Robertson & Zaragoza): per-term inverse document frequency,
     term-frequency saturation (k1), and document-length normalisation (b).
  3. Add precise structural field-match boosts on top (the issue naming this exact
     file path; a dotted module path resolving to this file): the standard
     BM25-retrieval pattern of layering exact-field boosts over the BM25 core.
  4. Return top-N highest-scoring paths.

The point is to be useful, not optimal. A moderately-good ranker is enough to
test the hypothesis that front-loading file hints cuts exploration turns.
"""
from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Token extraction ─────────────────────────────────────────────────────────

# Code identifiers: CamelCase, snake_case, ALL_CAPS — length >= 3.
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

# Dotted Python paths, e.g. django.db.models.fields.Field (>= 2 dots OR
# >= 2 segments and ends in a capitalised name).
_DOTTED_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){1,})\b")

# Inline-quoted file paths, e.g. 'src/foo.py' or `bar/baz.py`. Catches the most
# common forms; not exhaustive.
_PATH_RE = re.compile(
    r"""(?<![A-Za-z0-9])  # not preceded by an identifier char
        ((?:[A-Za-z0-9_-]+/){0,5}[A-Za-z0-9_-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cc|cpp|h|hpp|md|yml|yaml|toml|json))
        (?![A-Za-z0-9])""",
    re.VERBOSE,
)

# Common English words we never want to rank on. Short list; keep additions
# domain-neutral (no python/django/swe-specific terms).
_STOPWORDS = {
    "the", "and", "for", "with", "but", "not", "you", "are", "this", "that",
    "from", "have", "has", "had", "would", "could", "should", "will", "shall",
    "can", "may", "might", "must", "been", "being", "was", "were", "is", "am",
    "be", "to", "of", "in", "on", "at", "by", "as", "an", "a", "or", "if",
    "then", "than", "when", "where", "what", "which", "who", "how", "why",
    "do", "does", "did", "done", "all", "any", "some", "no", "yes", "out",
    "one", "two", "three", "use", "used", "uses", "using", "see", "also",
    "new", "old", "just", "only", "still", "even", "now", "here", "there",
}


@dataclass
class RankedFile:
    path: str
    score: float
    matched_tokens: tuple[str, ...]  # for debugging / explainability


def _extract_signals(issue_text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (paths, dotted_paths, identifiers) extracted from the issue text."""
    paths = _PATH_RE.findall(issue_text)
    dotted = _DOTTED_RE.findall(issue_text)
    # Identifiers: deduped, drop stopwords + drop those already inside a dotted path
    dotted_components = set()
    for d in dotted:
        dotted_components.update(d.split("."))
    raw_idents = _IDENT_RE.findall(issue_text)
    idents = []
    seen = set()
    for tok in raw_idents:
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if tok in dotted_components and len(tok) <= 4:
            # Short module segments (db, util, etc) bring too much noise on their own
            continue
        if low in seen:
            continue
        seen.add(low)
        idents.append(tok)
    return paths, dotted, idents


def _list_tracked_files(repo_root: Path,
                        extensions: Optional[tuple[str, ...]] = None,
                        timeout: int = 10) -> list[Path]:
    """Return relative paths of tracked files (filtered by extension)."""
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root, capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return []
    if res.returncode != 0:
        return []
    paths = [Path(p) for p in res.stdout.splitlines() if p.strip()]
    if extensions:
        paths = [p for p in paths if p.suffix.lower().lstrip(".") in extensions]
    return paths


def _file_peek(repo_root: Path, rel_path: Path, max_bytes: int = 2048) -> str:
    """Read the first max_bytes of a file. Returns '' on error / binary."""
    try:
        with (repo_root / rel_path).open("rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return ""
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _module_path_candidates(dotted: str) -> list[str]:
    """Given 'django.db.models.fields', return likely file paths."""
    parts = dotted.split(".")
    # Try increasingly-shallow prefixes; the deepest path is most specific.
    out = []
    for n in range(len(parts), 1, -1):
        head = "/".join(parts[:n])
        out.append(f"{head}.py")
        out.append(f"{head}/__init__.py")
    return out


# ── Okapi BM25 parameters + tokenisation ─────────────────────────────────────
# A token is a lowercased identifier (CamelCase / snake_case / ALL_CAPS, len>=3),
# stopwords removed. A file's document is its path tokens plus its first-2 KB peek
# tokens; the query is the identifiers extracted from the issue.

_BM25_K1 = 1.5             # term-frequency saturation
_BM25_B = 0.75            # document-length normalisation
_EXACT_PATH_BOOST = 10.0  # the issue named this exact file path
_MODULE_PATH_BOOST = 6.0  # a dotted module path resolves to this file
_PATH_SPLIT_RE = re.compile(r"[/._\-]+")


def _tokenize(text: str) -> list[str]:
    """Lowercased identifier tokens (len>=3, stopwords removed) -- the BM25 unit."""
    out = []
    for tok in _IDENT_RE.findall(text):
        low = tok.lower()
        if len(low) >= 3 and low not in _STOPWORDS:
            out.append(low)
    return out


def _path_tokens(path_str: str) -> list[str]:
    """Tokens from a file path (split on / . _ -), so a query identifier matching
    a path component contributes through BM25 as well as the exact-path boost."""
    return [t.lower() for t in _PATH_SPLIT_RE.split(path_str)
            if len(t) >= 3 and t.lower() not in _STOPWORDS]


def rank_relevant_files(
    repo_root: Path,
    issue_text: str,
    top_n: int = 10,
    extensions: tuple[str, ...] = ("py",),
) -> list[RankedFile]:
    """
    Score each tracked file against the issue text with Okapi BM25 plus exact
    path / module-path field-match boosts. Returns top-N by score (highest
    first), ties broken by alphabetical path.

    Scoring:
      BM25(query, document)  -- query = issue identifiers; document = the file's
            path tokens + first-2 KB peek tokens. IDF(t) = ln(1 + (N - df + 0.5) /
            (df + 0.5)); TF saturation with k1=1.5; length normalisation with
            b=0.75 against the corpus average document length.
      +10.0 exact path match  (the issue named 'src/foo.py' and we have it)
      +6.0  module-path match  (the issue named 'django.db.fields' and we have
            django/db/fields.py) -- standard BM25 exact-field boosting.
    """
    paths_in_text, dotted_in_text, idents = _extract_signals(issue_text)
    files = _list_tracked_files(repo_root, extensions=extensions)
    if not files:
        return []

    # Pre-compute module-path candidates + exact-path targets (the boost fields).
    module_targets = set()
    for d in dotted_in_text:
        module_targets.update(_module_path_candidates(d))
    path_targets = {p.lstrip("./") for p in paths_in_text}

    # The BM25 query: deduped lowercased issue identifiers.
    query: list[str] = []
    seen_q: set[str] = set()
    for tok in idents:
        low = tok.lower()
        if len(low) >= 3 and low not in _STOPWORDS and low not in seen_q:
            seen_q.add(low)
            query.append(low)

    # Build the corpus: one document (term-frequency map + length) per file, and
    # the document-frequency table over the whole corpus.
    docs: list[tuple[Path, dict[str, int], int]] = []
    df: dict[str, int] = {}
    total_len = 0
    for f in files:
        toks = _path_tokens(f.as_posix()) + _tokenize(_file_peek(repo_root, f))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        docs.append((f, tf, len(toks)))
        total_len += len(toks)
        for t in tf:
            df[t] = df.get(t, 0) + 1

    n_docs = len(docs)
    avgdl = (total_len / n_docs) if n_docs else 0.0

    def _idf(term: str) -> float:
        n = df.get(term, 0)
        return math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    q_idf = {t: _idf(t) for t in query}

    ranked: list[RankedFile] = []
    for f, tf, dl in docs:
        s = f.as_posix()
        score = 0.0
        matched: list[str] = []

        # BM25 content score over the query terms.
        len_norm = (1.0 - _BM25_B + _BM25_B * (dl / avgdl)) if avgdl else 1.0
        for t in query:
            f_tf = tf.get(t, 0)
            if f_tf == 0:
                continue
            score += q_idf[t] * (f_tf * (_BM25_K1 + 1.0)) / (f_tf + _BM25_K1 * len_norm)
            matched.append(f"bm25:{t}")

        # Exact path / module-path field-match boosts.
        if s in path_targets or any(s.endswith("/" + p) or s == p for p in path_targets):
            score += _EXACT_PATH_BOOST
            matched.append(f"path:{s}")
        if s in module_targets:
            score += _MODULE_PATH_BOOST
            matched.append(f"module:{s}")

        if score > 0.0:
            ranked.append(RankedFile(path=s, score=round(score, 4),
                                     matched_tokens=tuple(matched)))

    # Sort by score desc, then path asc for stable ordering
    ranked.sort(key=lambda r: (-r.score, r.path))
    return ranked[:top_n]


def format_for_prompt(ranked: list[RankedFile]) -> str:
    """Render the top-N as a markdown block suitable for prepending to a prompt."""
    if not ranked:
        return ""
    lines = [
        "## Likely relevant files",
        "",
        "Based on the issue text, these repo files look most relevant. "
        "Start your investigation here. If none of these are right, "
        "search the repo yourself.",
        "",
    ]
    for r in ranked:
        lines.append(f"- `{r.path}`")
    lines.append("")
    return "\n".join(lines)
