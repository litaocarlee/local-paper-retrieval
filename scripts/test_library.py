#!/usr/bin/env python3

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import library


class LibraryLogicTest(unittest.TestCase):
    def test_json_extraction_and_query_tokens(self) -> None:
        self.assertEqual(library.extract_json_object('```json\n{"queries": ["Co-12"]}\n```')["queries"], ["Co-12"])
        expression = library.fts_expression("Co-12 ICU graph explanations")
        self.assertIn('"Co-12"', expression)
        self.assertIn('"explanations"', expression)
        self.assertEqual(
            list(library.nested_strings({"exact": ["Co-12"], "synonyms": ["XAI evaluation"]})),
            ["Co-12", "XAI evaluation"],
        )

    def test_rrf_and_excerpt(self) -> None:
        ordered, scores, sources = library.reciprocal_rank_fusion(
            [("fts", ["a", "b"]), ("dense", ["b", "c"])]
        )
        self.assertEqual(ordered[0], "b")
        self.assertGreater(scores["b"], scores["a"])
        self.assertEqual(len(sources["b"]), 2)
        excerpt = library.best_excerpt("x " * 1000 + "target method" + " y" * 1000, ["target method"], 200)
        self.assertIn("target method", excerpt)

    def test_manifest_and_visual_gate(self) -> None:
        indexed = {
            "same.pdf": {"size_bytes": 10, "mtime_ns": 20},
            "changed.pdf": {"size_bytes": 10, "mtime_ns": 20},
            "removed.pdf": {"size_bytes": 10, "mtime_ns": 20},
        }
        snapshot = {
            "same.pdf": (Path("same.pdf"), 10, 20),
            "changed.pdf": (Path("changed.pdf"), 11, 20),
            "added.pdf": (Path("added.pdf"), 10, 20),
        }
        self.assertEqual(
            library.manifest_changes(indexed, snapshot),
            {"changed": True, "added_paths": 1, "removed_paths": 1, "modified_paths": 1},
        )
        self.assertFalse(library.visual_review_gate("Summarize the method", "See Figure 2")["required"])
        self.assertTrue(library.visual_review_gate("Compare the performance", "Table 2 reports results")["required"])
        self.assertTrue(library.visual_review_gate("What does Figure 2 show?", "plain text")["required"])

    def test_jev_choice_parse_and_sort(self) -> None:
        candidates = [
            {"id": "page-a", "rrf_score": 0.1, "title": "A", "pdf_page": 1, "excerpt": "a"},
            {"id": "page-b", "rrf_score": 0.9, "title": "B", "pdf_page": 2, "excerpt": "b"},
            {"id": "page-c", "rrf_score": 0.5, "title": "C", "pdf_page": 3, "excerpt": "c"},
        ]
        answers = {
            "best": {
                "type": "choice",
                "choice": "c1",
                "confidence": 0.8,
                "probabilities": {"c0": 0.1, "c1": 0.7, "c2": 0.15, "none": 0.05},
            },
            "fit_c1": {"type": "noul", "noul": 0.92},
            "grade_c1": {"type": "score", "score": 2.4},
        }
        judgment = library.parse_jev_judgment(candidates, answers, include_noul=True, include_score=True)
        self.assertEqual(judgment["choice"], "page-b")
        self.assertEqual(judgment["by_id"]["page-b"]["noul"], 0.92)
        self.assertEqual(judgment["by_id"]["page-b"]["score"], 2.4)
        ordered = sorted(candidates, key=lambda item: library.choice_sort_key(item, judgment))
        self.assertEqual([item["id"] for item in ordered], ["page-b", "page-c", "page-a"])
        none_judgment = library.parse_jev_choice(
            candidates,
            {"best": {"type": "choice", "choice": "none", "probabilities": {"none": 0.9, "c0": 0.05, "c1": 0.03, "c2": 0.02}}},
        )
        none_order = sorted(candidates, key=lambda item: library.choice_sort_key(item, none_judgment))
        self.assertEqual([item["id"] for item in none_order], ["page-b", "page-c", "page-a"])
        llm_ranked = sorted(
            candidates,
            key=lambda item: library.rerank_sort_key(
                item,
                {"page-a": {"relevance": 3}, "page-c": {"relevance": 1}},
            ),
        )
        self.assertEqual([item["id"] for item in llm_ranked], ["page-a", "page-c", "page-b"])

    def test_schema_v1_migrates_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO meta VALUES ('schema_version', '1')")
            connection.commit()
            connection.close()
            migrated = library.connect(path)
            self.assertEqual(migrated.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0], "2")
            self.assertIsNotNone(
                migrated.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_state'").fetchone()
            )
            migrated.close()


if __name__ == "__main__":
    unittest.main()
