---
name: private-kb-retrieve
description: Agent skill that finds a page in a local PDF library and verifies it on the original PDF. Uses SHA-256 deduplication, SQLite FTS5, stored page vectors, LLM query expansion, TypeSafe Jev Choice with optional Noul/Score and an LLM fallback, and conditional visual review. Use when answering a question from local full text or retrieving exact evidence with a PDF page anchor.
---

# Private KB retrieve

Agent skill. Treat the PDF as the evidence source. Treat index snippets, embedding matches, LLM rankings, and Jev judgments only as candidate locators.

Never print or persist API keys. Do not build a second index.

## Quick start

```bash
python scripts/library.py search \
  "Which paper evaluates graph explanations?"

python scripts/library.py read \
  <doc_id> --page <pdf_page> --context 1
```

Do not run `sync` as a first step. `search` already compares the current PDF manifest with the index and refreshes only what changed.

## Workflow

1. Run `search`. Sync hashes only new or modified PDFs; use SHA-256 to deduplicate content.
2. Read the `actors`, `jev`, and `timings_seconds` fields. The language model turns the question into a few short English queries. SQLite FTS5 retrieves pages by those queries. The embedding model embeds the queries and compares them with page vectors already stored in the index. Python fuses the two rankings with reciprocal rank fusion. Jev then runs a Choice over the candidate excerpts plus `none`. If TypeSafe is missing or fails, the language model remains the rerank fallback.
   Default search is Choice only. Add `--jev-noul` when pages must be kept or dropped by suitability, and `--jev-score` when each page needs a graded evidence-strength. If both might be useful, pass both flags in the same `search`; they share one TypeSafe call. Do not add a prior Jev call to decide which flags to use.
   Jev returns probabilities, not evidence. If `jev.choice` is `none` or `jev.confidence` is low, do not treat rank 1 as settled; `read` more than one page or rerun with `--jev-noul` / `--jev-score`.
3. Select the smallest set of candidate pages that can answer the question.
4. Run `read` for each selected page. It checks that the PDF still matches the index and re-extracts the original page. SHA-256 is recomputed only after a manifest mismatch or with `--force-hash`.
5. Inspect `visual_verification.required`. Render and visually inspect only pages selected by the cheap gate because the question explicitly targets a visual, a numeric claim is tied to a table or figure, or extracted text quality is weak.
6. Build an evidence card before writing any factual claim.

## Evidence card contract

Record:

- `claim`
- `support_status`: `direct`, `partial`, or `not_found`
- `doc_id` and title
- `pdf_page`
- section or table/figure label when visible
- a short exact excerpt or a clearly marked paraphrase
- boundary conditions needed to avoid overstating the source

Do not turn a Jev Choice, noul, score, or an LLM rerank reason into evidence. Do not cite a cached snippet without `read` verification. Label synthesis across papers as inference.

## Retrieval boundaries

- Prefer an exact DOI, title, author, or method-name hit before dense retrieval.
- Keep dense retrieval as a supplement to FTS5; the sparse path remains a valid fallback when either API is unavailable.
- Send only the user query to query expansion. Post-retrieve judging may send short candidate excerpts, never full PDFs or full pages. Jev Choice (optional Noul/Score) judges those excerpts after reciprocal rank fusion; it does not search the library.
- Index PDFs as primary evidence. Treat local HTML reports and BibTeX files as secondary metadata unless the user explicitly asks for them.
- Search the web only after local coverage is exhausted or when current citation metadata must be verified. Distinguish local and external evidence.

## Deterministic modes

Use `search --no-llm` to test raw FTS5 and dense retrieval without query expansion or Jev, `--no-dense` to test sparse retrieval, and `--no-rerank` to keep fused ordering after expansion. Use `search --jev-noul` and/or `--jev-score` only when the calling agent needs those extra judgments. Use `sync --no-embeddings` only when intentionally building a sparse-only index.

Use `read --force-hash` only for an explicit full-integrity check. Report measured stage timings as observations from the current run, not universal latency guarantees.

If an optional language-model, TypeSafe, or query-embedding call fails, report the degraded mode and continue with the remaining index. If PDF extraction, SHA verification, database schema, or an explicitly requested embedding build fails, stop and fix the contract.
