---
name: private-kb-retrieve
description: Finds a page in a private PDF library and verifies the answer on the original PDF. Use when the user asks about local papers, a private knowledge base, or wants a page-level citation instead of a web result. Full-text search and stored page vectors locate the page; Jev only ranks excerpts.
---

# Private KB retrieve

The PDF is the evidence. Index snippets, embedding matches, and Jev judgments only locate a page.

Never print or persist API keys. Do not build a second index.

## When to apply

- The question is about PDFs already in this library
- The user wants a page, passage, or citation from those PDFs
- A claim must be checked against the original page

## Quick start

From the skill root. Do not run `sync` first.

```bash
python scripts/library.py search "<question>"
python scripts/library.py read <doc_id> --page <pdf_page> --context 1 --query "<question>"
```

## Workflow

1. Run `search` on the user's question.
2. Read `actors`, `jev`, and `timings_seconds`. If the language model, embeddings, or Jev failed, say which stage degraded and continue with what remains.
3. If `jev.choice` is `none` or confidence is low, do not treat rank 1 as settled. Read more than one page, or rerun once with `--jev-noul` and `--jev-score` together. Do not make a separate call only to decide which flag to pass.
4. Pick the smallest set of pages that can answer the question.
5. `read` each selected page. This re-extracts the original PDF. SHA-256 runs only when the file manifest mismatches, or with `--force-hash`.
6. If `visual_verification.required` is true, render that page and look at it. The gate fires when the question targets a figure, table, equation, or number, or when the extracted text is weak.
7. Write an evidence card before any factual claim.

## Evidence card

```markdown
- claim:
- support_status: direct | partial | not_found
- doc_id:
- title:
- pdf_page:
- section or figure:
- excerpt:
- boundary:
```

A Jev Choice, Noul, Score, or rerank reason is not evidence. Do not cite an index snippet that `read` has not confirmed. Mark a synthesis across papers as inference.

## Flags

- `--jev-noul` keeps or drops each page by suitability. `--jev-score` grades evidence strength. Pass both in the same `search` when both are needed.
- `--no-llm` skips query expansion and Jev. `--no-dense` skips vectors. `--no-rerank` keeps fused order.
- `--force-hash` recomputes SHA-256 even when the file manifest matches. `sync --no-embeddings` builds a text-only index.
- Report `timings_seconds` from that run only.

## Rules

- Retrieve pages, not whole documents. Keep at most the pages `search` returns.
- Prefer an exact DOI, title, author, or method name when the question already contains one.
- Send the user question to query expansion. Send Jev only short excerpts, never a full PDF or a full page.
- Index PDFs. Use local HTML or BibTeX only when the user asks for that metadata.
- Search the web only after the local library has no usable page, or when a citation's publication details must be checked. Label web results as external.
- If PDF extraction, SHA verification, the database schema, or an explicitly requested embedding build fails, stop. Do not present a partial index as complete.
