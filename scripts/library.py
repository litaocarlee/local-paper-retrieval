#!/usr/bin/env python3
"""Hybrid local-PDF retrieval with original-page verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shlex
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import dotenv_values
from openai import OpenAI


SCHEMA_VERSION = "2"
DEFAULT_EMBEDDING_DIM = 1024
RRF_K = 60
DEFAULT_LIBRARY = Path("papers")
DEFAULT_INDEX = Path(".cache") / "index.sqlite"
LOGGER = logging.getLogger("local-paper-retrieval")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
VISUAL_INTENT_RE = re.compile(
    r"\b(?:table|figure|fig\.?|plot|chart|diagram|equation)\b|表格|图表|图中|公式|示意图",
    re.IGNORECASE,
)
NUMERIC_INTENT_RE = re.compile(
    r"\b(?:value|number|metric|score|performance|coefficient|p[- ]?value|auc|accuracy|comparison)\b|"
    r"数值|多少|指标|性能|系数|显著性|比较",
    re.IGNORECASE,
)
VISUAL_MARKER_RE = re.compile(
    r"\b(?:table|figure|fig\.?|equation)\s*[A-Z]?\d+\b|表\s*\d+|图\s*\d+",
    re.IGNORECASE,
)
JEV_SCORE_LEVELS = [
    "The excerpt does not contain any fact, method, result, or definition needed to answer the question.",
    "The excerpt is only topically related; the needed evidence is not in this excerpt.",
    "The excerpt contains partial evidence but is missing a key constraint, metric, or definition.",
    "The excerpt itself states or shows the method, result, definition, or fact needed to answer the question.",
]


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve()
    for candidate in (cwd, *cwd.parents, *here.parents):
        if (candidate / "SKILL.md").is_file() and (candidate / "scripts" / "library.py").is_file():
            return candidate
    raise RuntimeError("Run this skill from its own directory")


def resolve_path(value: str | None, default: Path, root: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def require_command(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"Required command not found: {name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def pdf_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in run_text(["pdfinfo", str(path)]).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def extract_pages(path: Path, first: int | None = None, last: int | None = None) -> list[str]:
    args = ["pdftotext", "-layout", "-enc", "UTF-8"]
    if first is not None:
        args += ["-f", str(first)]
    if last is not None:
        args += ["-l", str(last)]
    args += [str(path), "-"]
    parts = run_text(args).split("\f")
    if parts and not parts[-1].strip():
        parts.pop()
    return parts


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def display_title(info: dict[str, str], path: Path) -> str:
    title = info.get("Title", "").strip()
    if not title or title.casefold() in {"untitled", "microsoft word - document1"}:
        return path.stem
    return re.sub(r"\s+", " ", title)


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def collection_for(path: Path, library: Path) -> str:
    relative = path.relative_to(library)
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def load_api_group(env_path: Path, prefix: str, required: bool = False) -> dict[str, str] | None:
    values = dotenv_values(env_path)
    group = {name: (values.get(f"{prefix}{name}") or "").strip() for name in ("url", "model", "key")}
    if not any(group.values()):
        if required:
            raise RuntimeError(f"Missing {prefix}url/{prefix}model/{prefix}key in {env_path}")
        return None
    if not all(group.values()):
        missing = [f"{prefix}{name}" for name, value in group.items() if not value]
        raise RuntimeError(f"Incomplete {prefix} API configuration: {', '.join(missing)}")
    return group


def load_typesafe_config(env_path: Path) -> dict[str, str] | None:
    values = dotenv_values(env_path)
    key = (values.get("TYPESAFE_API_KEY") or "").strip()
    if not key:
        return None
    return {
        "key": key,
        "base_url": (values.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").strip().rstrip("/"),
        "model": (values.get("TYPESAFE_DEFAULT_MODEL") or "jev-latest").strip() or "jev-latest",
    }


def api_client(group: dict[str, str], timeout: float = 60.0) -> OpenAI:
    return OpenAI(api_key=group["key"], base_url=group["url"], timeout=timeout)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            doi TEXT NOT NULL,
            canonical_path TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            collections_json TEXT NOT NULL,
            page_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pages (
            page_key TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            page_no INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            search_text TEXT NOT NULL,
            quality TEXT NOT NULL,
            embedding BLOB,
            embedding_dim INTEGER,
            embedding_model TEXT,
            UNIQUE(doc_id, page_no)
        );
        CREATE TABLE IF NOT EXISTS file_state (
            path TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            page_key UNINDEXED,
            doc_id UNINDEXED,
            page_no UNINDEXED,
            title,
            paths,
            collections,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    version = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if version and version[0] not in {"1", SCHEMA_VERSION}:
        raise RuntimeError(f"Index schema {version[0]} != expected {SCHEMA_VERSION}; rebuild the derived database")
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    connection.commit()
    return connection


def pdf_snapshot(library: Path, root: Path) -> dict[str, tuple[Path, int, int]]:
    snapshot: dict[str, tuple[Path, int, int]] = {}
    paths = sorted(
        (path for path in library.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=str,
    )
    for path in paths:
        stat = path.stat()
        snapshot[relative_or_absolute(path, root)] = (path, stat.st_size, stat.st_mtime_ns)
    return snapshot


def indexed_file_states(connection: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {
        row["path"]: row
        for row in connection.execute("SELECT path, doc_id, size_bytes, mtime_ns FROM file_state")
    }


def manifest_changes(
    indexed: dict[str, sqlite3.Row],
    snapshot: dict[str, tuple[Path, int, int]],
) -> dict[str, int | bool]:
    current_paths = set(snapshot)
    indexed_paths = set(indexed)
    modified = sum(
        (indexed[path]["size_bytes"], indexed[path]["mtime_ns"]) != snapshot[path][1:]
        for path in current_paths & indexed_paths
    )
    added = len(current_paths - indexed_paths)
    removed = len(indexed_paths - current_paths)
    return {
        "changed": bool(added or removed or modified),
        "added_paths": added,
        "removed_paths": removed,
        "modified_paths": modified,
    }


def elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def rebuild_fts(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM pages_fts")
    connection.execute(
        """
        INSERT INTO pages_fts(page_key, doc_id, page_no, title, paths, collections, text)
        SELECT p.page_key, p.doc_id, p.page_no, d.title, d.aliases_json,
               d.collections_json, p.search_text
        FROM pages p JOIN documents d ON d.doc_id = p.doc_id
        WHERE p.quality != 'empty'
        """
    )


def embed_pending_pages(
    connection: sqlite3.Connection,
    env_path: Path,
    model: str,
    dimensions: int,
    batch_size: int,
) -> int:
    group = load_api_group(env_path, "emb", required=True)
    assert group is not None
    if group["model"] != model:
        raise RuntimeError("Embedding model changed while sync was running")
    rows = connection.execute(
        """
        SELECT page_key, search_text FROM pages
        WHERE quality = 'ok' AND (
            embedding IS NULL OR embedding_model != ? OR embedding_dim != ?
        ) ORDER BY page_key
        """,
        (model, dimensions),
    ).fetchall()
    if not rows:
        return 0
    client = api_client(group, timeout=120.0)
    completed = 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        response = client.embeddings.create(
            model=model,
            input=[row["search_text"][:100_000] for row in batch],
            dimensions=dimensions,
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding API returned a different number of vectors than inputs")
        for row, vector in zip(batch, vectors, strict=True):
            array = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(array))
            if array.size != dimensions or not np.isfinite(array).all() or norm == 0:
                raise RuntimeError(f"Invalid embedding for {row['page_key']}")
            array /= norm
            connection.execute(
                "UPDATE pages SET embedding=?, embedding_dim=?, embedding_model=? WHERE page_key=?",
                (array.tobytes(), dimensions, model, row["page_key"]),
            )
        connection.commit()
        completed += len(batch)
        LOGGER.info("Embedded %d/%d pages", completed, len(rows))
    return completed


def command_sync(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    root = repo_root()
    library = resolve_path(args.library, root / DEFAULT_LIBRARY, root)
    db_path = resolve_path(args.db, root / DEFAULT_INDEX, root)
    env_path = resolve_path(args.env, root / ".env", root)
    if not library.is_dir():
        raise RuntimeError(f"Literature library not found: {library}")
    require_command("pdfinfo")
    require_command("pdftotext")

    stage_started = time.perf_counter()
    connection = connect(db_path)
    snapshot = pdf_snapshot(library, root)
    cached_files = indexed_file_states(connection)
    changes = manifest_changes(cached_files, snapshot)
    grouped: dict[str, list[Path]] = defaultdict(list)
    file_rows: list[tuple[str, str, int, int]] = []
    files_hashed = 0
    cached_hashes_reused = 0
    for path_key, (path, size_bytes, mtime_ns) in snapshot.items():
        cached = cached_files.get(path_key)
        if cached and (cached["size_bytes"], cached["mtime_ns"]) == (size_bytes, mtime_ns):
            doc_id = cached["doc_id"]
            cached_hashes_reused += 1
        else:
            doc_id = sha256_file(path)
            files_hashed += 1
        grouped[doc_id].append(path)
        file_rows.append((path_key, doc_id, size_bytes, mtime_ns))
    timings["file_scan_and_hash"] = elapsed_seconds(stage_started)

    stage_started = time.perf_counter()
    existing = {row[0] for row in connection.execute("SELECT doc_id FROM documents")}
    incoming = set(grouped)
    removed = existing - incoming
    new_documents = 0
    if changes["changed"]:
        for doc_id in removed:
            connection.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

        for index, (doc_id, aliases) in enumerate(sorted(grouped.items()), start=1):
            canonical = min(aliases, key=lambda path: (len(relative_or_absolute(path, root)), str(path).casefold()))
            alias_names = [relative_or_absolute(path, root) for path in aliases]
            collections = sorted({collection_for(path, library) for path in aliases})
            if doc_id not in existing:
                info = pdf_info(canonical)
                pages = extract_pages(canonical)
                expected_pages = int(info.get("Pages", len(pages)))
                if len(pages) < expected_pages:
                    pages.extend([""] * (expected_pages - len(pages)))
                pages = pages[:expected_pages]
                head = "\n".join(pages[:3])
                doi_match = DOI_RE.search(head)
                connection.execute(
                    """
                    INSERT INTO documents(
                        doc_id, title, authors, doi, canonical_path, aliases_json,
                        collections_json, page_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        display_title(info, canonical),
                        info.get("Author", ""),
                        doi_match.group(0).rstrip(".,;)") if doi_match else "",
                        relative_or_absolute(canonical, root),
                        json.dumps(alias_names, ensure_ascii=False),
                        json.dumps(collections, ensure_ascii=False),
                        expected_pages,
                    ),
                )
                for page_no, raw_text in enumerate(pages, start=1):
                    search_text = normalize_text(raw_text)
                    quality = "empty" if not search_text else ("low_text" if len(search_text) < 80 else "ok")
                    connection.execute(
                        """
                        INSERT INTO pages(page_key, doc_id, page_no, raw_text, search_text, quality)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (f"{doc_id}:{page_no}", doc_id, page_no, raw_text, search_text, quality),
                    )
                new_documents += 1
            else:
                connection.execute(
                    """
                    UPDATE documents SET canonical_path=?, aliases_json=?, collections_json=?
                    WHERE doc_id=?
                    """,
                    (
                        relative_or_absolute(canonical, root),
                        json.dumps(alias_names, ensure_ascii=False),
                        json.dumps(collections, ensure_ascii=False),
                        doc_id,
                    ),
                )
            if index % 10 == 0:
                LOGGER.info("Indexed metadata %d/%d documents", index, len(grouped))

        connection.execute("DELETE FROM file_state")
        connection.executemany(
            "INSERT INTO file_state(path, doc_id, size_bytes, mtime_ns) VALUES (?, ?, ?, ?)",
            file_rows,
        )
        rebuild_fts(connection)
        connection.commit()
    timings["metadata_and_fts"] = elapsed_seconds(stage_started)

    stage_started = time.perf_counter()
    embedded = 0
    embedding_model = None
    if not args.no_embeddings:
        group = load_api_group(env_path, "emb", required=True)
        assert group is not None
        embedding_model = group["model"]
        embedded = embed_pending_pages(
            connection,
            env_path,
            embedding_model,
            args.embedding_dim,
            args.batch_size,
        )
    timings["embedding"] = elapsed_seconds(stage_started)

    counts = connection.execute(
        """
        SELECT COUNT(*) AS pages,
               SUM(quality='ok') AS ok_pages,
               SUM(quality='low_text') AS low_text_pages,
               SUM(quality='empty') AS empty_pages,
               SUM(embedding IS NOT NULL) AS embedded_pages
        FROM pages
        """
    ).fetchone()
    connection.close()
    timings["total"] = elapsed_seconds(total_started)
    summary = {
        "schema_version": int(SCHEMA_VERSION),
        "library": relative_or_absolute(library, root),
        "database": relative_or_absolute(db_path, root),
        "pdf_paths": len(snapshot),
        "unique_documents": len(grouped),
        "new_documents": new_documents,
        "removed_documents": len(removed),
        "files_hashed": files_hashed,
        "cached_hashes_reused": cached_hashes_reused,
        "changes": changes,
        "pages": counts["pages"],
        "ok_pages": counts["ok_pages"],
        "low_text_pages": counts["low_text_pages"],
        "empty_pages": counts["empty_pages"],
        "embedded_pages": counts["embedded_pages"],
        "embeddings_added": embedded,
        "embedding_model": embedding_model,
        "embedding_dimension": args.embedding_dim if embedding_model else None,
        "timings_seconds": timings,
    }
    if not getattr(args, "quiet", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value


def chat_json(group: dict[str, str], system: str, user: str, max_tokens: int = 2048) -> dict[str, Any]:
    provider_options = (
        {"extra_body": {"thinking": {"type": "disabled"}}}
        if "deepseek" in group["model"].casefold() or "deepseek.com" in group["url"].casefold()
        else {}
    )
    response = api_client(group).chat.completions.create(
        model=group["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        **provider_options,
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(f"LLM JSON generation ended with {choice.finish_reason}")
    content = choice.message.content or ""
    return extract_json_object(content)


def unique_strings(values: Iterable[Any], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        clean = re.sub(r"\s+", " ", value).strip()
        folded = clean.casefold()
        if clean and folded not in seen:
            seen.add(folded)
            result.append(clean)
        if len(result) >= limit:
            break
    return result


def nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from nested_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from nested_strings(nested)


def expand_query(query: str, group: dict[str, str]) -> tuple[list[str], list[str], dict[str, Any]]:
    plan = chat_json(
        group,
        "You plan high-recall searches over English academic PDF pages. Return JSON only and do not answer the question.",
        json.dumps(
            {
                "question": query,
                "required_schema": {
                    "queries": ["2-5 concise English retrieval queries"],
                    "terms": ["exact methods, acronyms, entities, and synonyms"],
                },
                "rules": [
                    "Preserve named methods, metrics, datasets, authors, and numbers.",
                    "Translate non-English questions into field-standard English terminology.",
                    "Include both exact terminology and one conceptual paraphrase.",
                ],
            },
            ensure_ascii=False,
        ),
    )
    queries = unique_strings([query, *nested_strings(plan.get("queries", []))], limit=6)
    terms = unique_strings([*nested_strings(plan.get("terms", [])), *queries], limit=24)
    return queries, terms, plan


def fts_expression(query: str) -> str:
    tokens = unique_strings(re.findall(r"[^\W_]+(?:-[^\W_]+)*", query, flags=re.UNICODE), limit=24)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens if len(token) > 1)


def sparse_rankings(connection: sqlite3.Connection, queries: list[str], limit: int) -> list[tuple[str, list[str]]]:
    rankings: list[tuple[str, list[str]]] = []
    for index, query in enumerate(queries):
        expression = fts_expression(query)
        if not expression:
            continue
        rows = connection.execute(
            """
            SELECT page_key FROM pages_fts
            WHERE pages_fts MATCH ?
            ORDER BY bm25(pages_fts, 0, 0, 0, 8, 4, 2, 1)
            LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
        rankings.append((f"fts:{index}", [row["page_key"] for row in rows]))
    return rankings


def dense_rankings(
    connection: sqlite3.Connection,
    queries: list[str],
    env_path: Path,
    limit: int,
) -> list[tuple[str, list[str]]]:
    models = connection.execute(
        """
        SELECT DISTINCT embedding_model, embedding_dim FROM pages
        WHERE embedding IS NOT NULL
        """
    ).fetchall()
    if not models:
        return []
    if len(models) != 1:
        raise RuntimeError("Index contains multiple embedding contracts; rerun sync")
    model, dimensions = models[0]
    group = load_api_group(env_path, "emb", required=True)
    assert group is not None
    if group["model"] != model:
        raise RuntimeError("Configured embedding model differs from the indexed model; rerun sync")
    response = api_client(group, timeout=120.0).embeddings.create(
        model=model,
        input=queries,
        dimensions=dimensions,
    )
    query_vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    rows = connection.execute(
        "SELECT page_key, embedding FROM pages WHERE embedding IS NOT NULL ORDER BY page_key"
    ).fetchall()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    page_keys = [row["page_key"] for row in rows]
    rankings: list[tuple[str, list[str]]] = []
    for index, vector in enumerate(query_vectors):
        query_vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if query_vector.size != dimensions or not np.isfinite(query_vector).all() or norm == 0:
            raise RuntimeError("Embedding API returned an invalid query vector")
        query_vector /= norm
        order = np.argsort(matrix @ query_vector)[::-1][:limit]
        rankings.append((f"dense:{index}", [page_keys[position] for position in order]))
    return rankings


def reciprocal_rank_fusion(rankings: list[tuple[str, list[str]]]) -> tuple[list[str], dict[str, float], dict[str, list[str]]]:
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)
    for channel, ranking in rankings:
        for rank, page_key in enumerate(ranking, start=1):
            scores[page_key] += 1.0 / (RRF_K + rank)
            sources[page_key].append(f"{channel}@{rank}")
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return ordered, dict(scores), dict(sources)


def best_excerpt(text: str, terms: list[str], limit: int = 1600) -> str:
    clean = normalize_text(text)
    if len(clean) <= limit:
        return clean
    folded = clean.casefold()
    positions = [folded.find(term.casefold()) for term in terms if len(term) >= 3]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(clean), start + limit)
    start = max(0, end - limit)
    return clean[start:end]


def visual_review_gate(query: str, text: str, quality: str = "ok") -> dict[str, Any]:
    reasons: list[str] = []
    if quality != "ok":
        reasons.append(f"text_quality:{quality}")
    if VISUAL_INTENT_RE.search(query):
        reasons.append("explicit_visual_question")
    elif NUMERIC_INTENT_RE.search(query) and VISUAL_MARKER_RE.search(text):
        reasons.append("numeric_claim_in_table_or_figure")
    # ponytail: heuristic gate; add layout classification only after observed false negatives.
    return {"required": bool(reasons), "reasons": reasons}


def load_candidates(
    connection: sqlite3.Connection,
    ordered: list[str],
    scores: dict[str, float],
    sources: dict[str, list[str]],
    terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    per_document: dict[str, int] = defaultdict(int)
    for page_key in ordered:
        row = connection.execute(
            """
            SELECT p.page_key, p.doc_id, p.page_no, p.raw_text, p.quality, d.title, d.authors,
                   d.doi, d.canonical_path, d.aliases_json, d.collections_json
            FROM pages p JOIN documents d ON d.doc_id = p.doc_id
            WHERE p.page_key=?
            """,
            (page_key,),
        ).fetchone()
        if row is None or per_document[row["doc_id"]] >= 3:
            continue
        per_document[row["doc_id"]] += 1
        candidates.append(
            {
                "id": row["page_key"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "authors": row["authors"],
                "doi": row["doi"],
                "path": row["canonical_path"],
                "aliases": json.loads(row["aliases_json"]),
                "collections": json.loads(row["collections_json"]),
                "pdf_page": row["page_no"],
                "text_quality": row["quality"],
                "rrf_score": scores[page_key],
                "retrieval_sources": sources[page_key],
                "excerpt": best_excerpt(row["raw_text"], terms),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def typesafe_system_one(config: dict[str, str], state: Any, questions: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    payload = json.dumps({"state": state, "model": config["model"], "questions": questions}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f"{config['base_url']}/v1/systemone",
        data=payload,
        headers={"Authorization": f"Bearer {config['key']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"TypeSafe API returned HTTP {error.code}") from error


def candidate_slot(index: int) -> str:
    return f"c{index}"


def jev_state(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question": query,
        "candidates": [
            {
                "title": candidate["title"],
                "pdf_page": candidate["pdf_page"],
                "excerpt": candidate["excerpt"],
            }
            for candidate in candidates
        ],
    }


def jev_questions(candidates: list[dict[str, Any]], include_noul: bool, include_score: bool) -> dict[str, Any]:
    criteria = {
        candidate_slot(index): (
            f"Use `candidates[{index}]`. Title and page are in that object; judge only its excerpt."
        )
        for index in range(len(candidates))
    }
    criteria["none"] = "No candidate excerpt contains evidence that can answer the question."
    questions: dict[str, Any] = {
        "best": {
            "type": "choice",
            "instructions": (
                "Which candidate excerpt best answers `question`? "
                "Choose `none` if none of the excerpts contains usable evidence. "
                "Do not assume the needed evidence appears on another page."
            ),
            "criteria": criteria,
        }
    }
    for index in range(len(candidates)):
        slot = candidate_slot(index)
        if include_noul:
            questions[f"fit_{slot}"] = {
                "type": "noul",
                "instructions": (
                    f"Does `candidates[{index}].excerpt` itself contain evidence that can answer `question`? "
                    "Judge only this candidate."
                ),
                "criteria": {
                    "true": "The excerpt states or shows the method, result, definition, or fact needed to answer the question.",
                    "false": "The excerpt is only topically related, or the needed evidence is not in this excerpt.",
                },
            }
        if include_score:
            questions[f"grade_{slot}"] = {
                "type": "score",
                "instructions": (
                    f"How completely does `candidates[{index}].excerpt` itself answer `question`? "
                    "Judge only this candidate."
                ),
                "criteria": list(JEV_SCORE_LEVELS),
            }
    return questions


def parse_unit_interval(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= number <= 1.0:
        return number
    return None


def parse_jev_choice(candidates: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    answer = answers.get("best")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise RuntimeError("TypeSafe judge returned no usable Choice answer")
    slots = {candidate_slot(index): candidate["id"] for index, candidate in enumerate(candidates)}
    raw_choice = str(answer.get("choice", ""))
    if raw_choice != "none" and raw_choice not in slots:
        raise RuntimeError("TypeSafe Choice selected an unknown candidate")
    raw_probabilities = answer.get("probabilities") or {}
    probabilities = {"none": parse_unit_interval(raw_probabilities.get("none")) or 0.0}
    for slot, page_id in slots.items():
        probabilities[page_id] = parse_unit_interval(raw_probabilities.get(slot)) or 0.0
    confidence = parse_unit_interval(answer.get("confidence"))
    return {
        "choice": "none" if raw_choice == "none" else slots[raw_choice],
        "choice_slot": raw_choice,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def parse_jev_noul(candidates: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        answer = answers.get(f"fit_{candidate_slot(index)}")
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            continue
        noul = parse_unit_interval(answer.get("noul"))
        if noul is not None:
            result[candidate["id"]] = noul
    return result


def parse_jev_score(candidates: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    maximum = float(len(JEV_SCORE_LEVELS) - 1)
    for index, candidate in enumerate(candidates):
        answer = answers.get(f"grade_{candidate_slot(index)}")
        if not isinstance(answer, dict) or answer.get("type") != "score":
            continue
        try:
            score = float(answer["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= score <= maximum:
            result[candidate["id"]] = score
    return result


def parse_jev_judgment(
    candidates: list[dict[str, Any]],
    answers: dict[str, Any],
    include_noul: bool,
    include_score: bool,
) -> dict[str, Any]:
    choice = parse_jev_choice(candidates, answers)
    nouls = parse_jev_noul(candidates, answers) if include_noul else {}
    scores = parse_jev_score(candidates, answers) if include_score else {}
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        page_id = candidate["id"]
        item: dict[str, Any] = {
            "choice_probability": choice["probabilities"].get(page_id, 0.0),
            "is_winner": choice["choice"] == page_id,
        }
        if page_id in nouls:
            item["noul"] = nouls[page_id]
        if page_id in scores:
            item["score"] = scores[page_id]
        by_id[page_id] = item
    return {**choice, "noul_used": bool(nouls), "score_used": bool(scores), "by_id": by_id}


def jev_judge(
    query: str,
    candidates: list[dict[str, Any]],
    config: dict[str, str],
    include_noul: bool = False,
    include_score: bool = False,
) -> tuple[dict[str, Any], str]:
    response = typesafe_system_one(
        config,
        jev_state(query, candidates),
        jev_questions(candidates, include_noul, include_score),
    )
    return (
        parse_jev_judgment(candidates, response.get("answers") or {}, include_noul, include_score),
        str(response.get("model") or config["model"]),
    )


def choice_sort_key(candidate: dict[str, Any], judgment: dict[str, Any]) -> tuple:
    if judgment.get("choice") == "none":
        return (0, -candidate["rrf_score"], candidate["id"])
    probability = float(judgment.get("probabilities", {}).get(candidate["id"], 0.0))
    return (-probability, -candidate["rrf_score"], candidate["id"])


def rerank_sort_key(candidate: dict[str, Any], rerank_values: dict[str, dict[str, Any]]) -> tuple:
    item = rerank_values.get(candidate["id"], {})
    try:
        relevance = int(item.get("relevance", -1))
    except (TypeError, ValueError):
        relevance = -1
    primary = relevance / 3.0 if relevance >= 0 else -1.0
    return (-primary, -candidate["rrf_score"], candidate["id"])


def rerank(query: str, candidates: list[dict[str, Any]], group: dict[str, str]) -> dict[str, dict[str, Any]]:
    payload = [
        {
            "id": candidate["id"],
            "title": candidate["title"],
            "pdf_page": candidate["pdf_page"],
            "excerpt": candidate["excerpt"],
        }
        for candidate in candidates
    ]
    value = chat_json(
        group,
        "Rank candidate PDF pages by evidence explicitly present in each excerpt. Return JSON only. "
        "Use relevance 3 for direct evidence, 2 for partial evidence, 1 for context only, and 0 for no evidence. "
        "Return at most 12 candidates with relevance above 0, omit irrelevant candidates, and keep each reason under 20 words. "
        "Never raise a score because evidence likely appears elsewhere. Do not answer the question or invent evidence.",
        json.dumps(
            {
                "question": query,
                "candidates": payload,
                "required_schema": {
                    "ranking": [
                        {"id": "candidate id", "relevance": "integer 0-3", "reason": "brief evidence-fit reason"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        max_tokens=3072,
    )
    valid_ids = {candidate["id"] for candidate in candidates}
    result: dict[str, dict[str, Any]] = {}
    for item in value.get("ranking", []):
        if not isinstance(item, dict) or item.get("id") not in valid_ids:
            continue
        relevance = item.get("relevance", 0)
        try:
            relevance = max(0, min(3, int(relevance)))
        except (TypeError, ValueError):
            relevance = 0
        result[item["id"]] = {"relevance": relevance, "reason": str(item.get("reason", ""))[:240]}
    return result


def command_search(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    root = repo_root()
    library = resolve_path(args.library, root / DEFAULT_LIBRARY, root)
    db_path = resolve_path(args.db, root / DEFAULT_INDEX, root)
    env_path = resolve_path(args.env, root / ".env", root)
    if not library.is_dir():
        raise RuntimeError(f"Literature library not found: {library}")

    stage_started = time.perf_counter()
    connection = connect(db_path)
    snapshot = pdf_snapshot(library, root)
    changes = manifest_changes(indexed_file_states(connection), snapshot)
    timings["library_preflight"] = elapsed_seconds(stage_started)
    auto_sync = None
    if changes["changed"]:
        connection.close()
        stage_started = time.perf_counter()
        auto_sync = command_sync(
            argparse.Namespace(
                library=None,
                db=args.db,
                env=args.env,
                no_embeddings=False,
                embedding_dim=DEFAULT_EMBEDDING_DIM,
                batch_size=32,
                quiet=True,
            )
        )
        timings["auto_sync"] = elapsed_seconds(stage_started)
        connection = connect(db_path)
    else:
        timings["auto_sync"] = 0.0

    degraded: list[str] = []
    queries, terms, plan = [args.query], [args.query], {"queries": [args.query], "terms": [args.query]}
    llm_group = None if args.no_llm else load_api_group(env_path, "llm")
    query_expansion_used = False
    stage_started = time.perf_counter()
    if llm_group:
        try:
            queries, terms, plan = expand_query(args.query, llm_group)
            query_expansion_used = True
        except Exception as error:
            degraded.append(f"query_expansion:{type(error).__name__}")
            LOGGER.warning("LLM query expansion failed; using the original query")
    timings["llm_query_expansion"] = elapsed_seconds(stage_started)

    stage_started = time.perf_counter()
    rankings = sparse_rankings(connection, queries, args.retrieval_limit)
    timings["fts5"] = elapsed_seconds(stage_started)
    dense_used = False
    embedding_model = None
    stage_started = time.perf_counter()
    if not args.no_dense:
        try:
            dense = dense_rankings(connection, queries, env_path, args.retrieval_limit)
            rankings.extend(dense)
            dense_used = bool(dense)
            if dense_used:
                embedding_row = connection.execute(
                    "SELECT embedding_model FROM pages WHERE embedding IS NOT NULL LIMIT 1"
                ).fetchone()
                embedding_model = embedding_row[0] if embedding_row else None
            if not dense:
                degraded.append("dense:no_indexed_embeddings")
        except Exception as error:
            degraded.append(f"dense:{type(error).__name__}")
            LOGGER.warning("Dense retrieval failed; continuing with FTS5")
    timings["dense_retrieval"] = elapsed_seconds(stage_started)

    stage_started = time.perf_counter()
    ordered, scores, sources = reciprocal_rank_fusion(rankings)
    candidates = load_candidates(
        connection,
        ordered,
        scores,
        sources,
        terms,
        max(args.candidate_limit, args.top),
    )
    timings["rrf_and_candidate_load"] = elapsed_seconds(stage_started)
    rerank_values: dict[str, dict[str, Any]] = {}
    jev_judgment: dict[str, Any] | None = None
    rerank_used = False
    rerank_actor = None
    typesafe_choice_used = False
    stage_started = time.perf_counter()
    if not args.no_llm and not args.no_rerank and candidates:
        shortlist = candidates[:12]
        typesafe = load_typesafe_config(env_path)
        if typesafe:
            try:
                jev_judgment, rerank_actor = jev_judge(
                    args.query,
                    shortlist,
                    typesafe,
                    include_noul=args.jev_noul,
                    include_score=args.jev_score,
                )
                typesafe_choice_used = True
                candidates.sort(key=lambda candidate: choice_sort_key(candidate, jev_judgment))
                rerank_used = True
            except Exception as error:
                jev_judgment = None
                degraded.append(f"jev_choice:{type(error).__name__}")
                LOGGER.warning("TypeSafe Choice judge failed; falling back: %s", error)
        if not rerank_used and llm_group:
            try:
                rerank_values = rerank(args.query, shortlist, llm_group)
                rerank_actor = llm_group["model"]
                candidates.sort(key=lambda candidate: rerank_sort_key(candidate, rerank_values))
                rerank_used = bool(rerank_values)
            except Exception as error:
                degraded.append(f"rerank:{type(error).__name__}")
                LOGGER.warning("LLM reranking failed; retaining RRF order: %s", error)
    timings["llm_rerank"] = elapsed_seconds(stage_started)
    connection.close()

    results = []
    for rank, candidate in enumerate(candidates[: args.top], start=1):
        item = dict(candidate)
        item["rank"] = rank
        item["rerank"] = rerank_values.get(candidate["id"])
        if jev_judgment:
            item["jev"] = jev_judgment["by_id"].get(candidate["id"])
        item["visual_verification"] = visual_review_gate(
            args.query,
            candidate["excerpt"],
            candidate["text_quality"],
        )
        item["verification"] = {
            "required": True,
            "command": (
                f"library.py read {candidate['doc_id']} --page {candidate['pdf_page']} --context 1 "
                f"--query {shlex.quote(args.query)}"
            ),
        }
        results.append(item)
    timings["total"] = elapsed_seconds(total_started)
    print(
        json.dumps(
            {
                "schema_version": int(SCHEMA_VERSION),
                "query": args.query,
                "query_plan": plan,
                "queries_used": queries,
                "library_changes": changes,
                "auto_sync": auto_sync,
                "fts5_used": True,
                "dense_used": dense_used,
                "llm_query_expansion_used": query_expansion_used,
                "llm_rerank_used": rerank_used,
                "typesafe_choice_used": typesafe_choice_used,
                "jev_noul_used": bool(jev_judgment and jev_judgment.get("noul_used")),
                "jev_score_used": bool(jev_judgment and jev_judgment.get("score_used")),
                "jev": None
                if jev_judgment is None
                else {
                    "choice": jev_judgment["choice"],
                    "choice_slot": jev_judgment["choice_slot"],
                    "confidence": jev_judgment["confidence"],
                    "probabilities": jev_judgment["probabilities"],
                },
                "actors": {
                    "query_expansion": llm_group["model"] if query_expansion_used else None,
                    "fts5": "SQLite FTS5 (local)",
                    "dense_embedding": embedding_model,
                    "fusion": "Python RRF (local)",
                    "rerank": rerank_actor,
                },
                "external_candidate_excerpts_sent": rerank_used,
                "degraded": degraded,
                "timings_seconds": timings,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def resolve_document(connection: sqlite3.Connection, identifier: str) -> sqlite3.Row:
    rows = connection.execute(
        "SELECT * FROM documents WHERE doc_id LIKE ? ORDER BY doc_id",
        (f"{identifier}%",),
    ).fetchall()
    if not rows:
        raise ValueError(f"Unknown document id: {identifier}")
    if len(rows) > 1:
        raise ValueError(f"Ambiguous document id prefix: {identifier}")
    return rows[0]


def command_read(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    root = repo_root()
    db_path = resolve_path(args.db, root / DEFAULT_INDEX, root)
    if not db_path.is_file():
        raise RuntimeError("Index not found; run sync first")
    connection = connect(db_path)
    document = resolve_document(connection, args.doc_id)
    path = Path(document["canonical_path"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise RuntimeError(f"Indexed PDF path is missing: {path}")
    if args.page > document["page_count"]:
        raise ValueError(f"PDF page {args.page} exceeds document length {document['page_count']}")

    stage_started = time.perf_counter()
    stat = path.stat()
    file_state = connection.execute(
        "SELECT doc_id, size_bytes, mtime_ns FROM file_state WHERE path=?",
        (document["canonical_path"],),
    ).fetchone()
    manifest_match = bool(
        file_state
        and file_state["doc_id"] == document["doc_id"]
        and (file_state["size_bytes"], file_state["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns)
    )
    actual_hash = None
    if args.force_hash or not manifest_match:
        actual_hash = sha256_file(path)
        if actual_hash != document["doc_id"]:
            connection.close()
            raise RuntimeError("PDF changed after indexing; run sync before verification")
        connection.execute(
            "INSERT OR REPLACE INTO file_state(path, doc_id, size_bytes, mtime_ns) VALUES (?, ?, ?, ?)",
            (document["canonical_path"], actual_hash, stat.st_size, stat.st_mtime_ns),
        )
        connection.commit()
    integrity_method = "sha256" if actual_hash else "manifest_stat"
    timings["integrity_check"] = elapsed_seconds(stage_started)

    start = max(1, args.page - args.context)
    end = min(document["page_count"], args.page + args.context)
    stage_started = time.perf_counter()
    pages = extract_pages(path, start, end)
    if len(pages) != end - start + 1:
        raise RuntimeError("Original PDF page extraction returned an unexpected page count")
    timings["pdf_page_extraction"] = elapsed_seconds(stage_started)
    combined_text = "\n".join(pages)
    normalized_text = normalize_text(combined_text)
    quality = "empty" if not normalized_text else ("low_text" if len(normalized_text) < 80 else "ok")
    visual_verification = visual_review_gate(args.query or "", combined_text, quality)
    connection.close()
    timings["total"] = elapsed_seconds(total_started)
    print(
        json.dumps(
            {
                "schema_version": int(SCHEMA_VERSION),
                "indexed_sha256": document["doc_id"],
                "verified_sha256": actual_hash,
                "integrity_check": {
                    "method": integrity_method,
                    "full_hash_computed": actual_hash is not None,
                },
                "title": document["title"],
                "authors": document["authors"],
                "doi": document["doi"],
                "path": document["canonical_path"],
                "aliases": json.loads(document["aliases_json"]),
                "requested_pdf_page": args.page,
                "pages": [
                    {"pdf_page": page_no, "text": text}
                    for page_no, text in zip(range(start, end + 1), pages, strict=False)
                ],
                "visual_verification": visual_verification,
                "timings_seconds": timings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite index path")
    parser.add_argument("--env", help="dotenv path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Incrementally index local PDFs")
    sync.add_argument("--library", help="PDF library root")
    sync.add_argument("--no-embeddings", action="store_true")
    sync.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    sync.add_argument("--batch-size", type=int, default=32)
    sync.set_defaults(func=command_sync)

    search = subparsers.add_parser("search", help="Run hybrid retrieval and post-retrieve Jev Choice")
    search.add_argument("query")
    search.add_argument("--library", help="PDF library root")
    search.add_argument("--top", type=int, default=5)
    search.add_argument("--candidate-limit", type=int, default=12)
    search.add_argument("--retrieval-limit", type=int, default=30)
    search.add_argument("--no-llm", action="store_true")
    search.add_argument("--no-dense", action="store_true")
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument("--jev-noul", action="store_true", help="Also ask per-candidate suitability Noul in the same TypeSafe call")
    search.add_argument("--jev-score", action="store_true", help="Also ask per-candidate evidence-strength Score in the same TypeSafe call")
    search.set_defaults(func=command_search)

    read = subparsers.add_parser("read", help="Verify and re-read original PDF pages")
    read.add_argument("doc_id")
    read.add_argument("--page", type=int, required=True)
    read.add_argument("--context", type=int, default=1)
    read.add_argument("--query", help="Original question for conditional visual-review gating")
    read.add_argument("--force-hash", action="store_true", help="Recompute SHA-256 even when the file manifest matches")
    read.set_defaults(func=command_read)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if getattr(args, "top", 1) < 1 or getattr(args, "candidate_limit", 1) < 1:
        raise ValueError("Result limits must be positive")
    if getattr(args, "batch_size", 1) < 1 or getattr(args, "embedding_dim", 1) < 1:
        raise ValueError("Embedding batch size and dimension must be positive")
    if getattr(args, "page", 1) < 1 or getattr(args, "context", 0) < 0:
        raise ValueError("PDF page must be positive and context must be non-negative")
    args.func(args)


if __name__ == "__main__":
    main()
