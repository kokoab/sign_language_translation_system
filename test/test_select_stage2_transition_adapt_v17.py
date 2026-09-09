"""Focused contracts for the offline transition-adaptation selector."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_stage2_transition_adapt_v17.py"
SPEC = importlib.util.spec_from_file_location("select_stage2_transition_adapt_v17", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def seed(seed: int, target: int, local: int = 6, exact: int = 9,
         contextual: int = 43, citizen: int = 328) -> dict:
    return {"arm": "no_stem", "seed": seed, "checkpoint": "model.pth",
            "checkpoint_sha256": "0" * 64, "history": "history.json",
            "validation": {"summary": {"target_edits": target, "target_tokens": 284,
                "local_edits": local, "local_tokens": 259, "exact_edits": exact,
                "exact_tokens": 24, "contextual_edits": contextual,
                "contextual_tokens": 254, "citizen_correct": citizen,
                "citizen_samples": 378}, "domains": {}, "examples": {}}}


class SelectTransitionAdaptTests(unittest.TestCase):
    def test_two_seed_rule_and_tie_favors_no_stem(self) -> None:
        records = [seed(1701, 542), seed(1702, 542)]
        records += [{**seed(1701, 542), "arm": "with_stem"},
                    {**seed(1702, 542), "arm": "with_stem"}]
        decision = MODULE.select(records)
        self.assertEqual("no_stem", decision["selected_arm"])
        self.assertFalse(MODULE.arm_summary(records[:1], "no_stem")["qualified"])

    def test_integer_citizen_and_exact_gate_boundaries(self) -> None:
        self.assertTrue(MODULE.gates(seed(1701, 542)["validation"]["summary"])["citizen"])
        self.assertFalse(MODULE.gates(seed(1701, 542, citizen=327)["validation"]["summary"])["citizen"])
        self.assertTrue(MODULE.gates(seed(1701, 542, exact=9)["validation"]["summary"])["exact"])
        self.assertFalse(MODULE.gates(seed(1701, 542, exact=10)["validation"]["summary"])["exact"])

    def test_edit_derivation_counts_inserted_repeat(self) -> None:
        edits = MODULE.edit_operations([1, 2], [1, 2, 2])
        self.assertEqual({"substitutions": 0, "deletions": 0, "insertions": 1,
                          "repeated_sign_errors": 1}, edits)

    def test_edit_derivation_counts_deleted_reference_repeat(self) -> None:
        edits = MODULE.edit_operations([1, 2, 2], [1, 2])
        self.assertEqual({"substitutions": 0, "deletions": 1, "insertions": 0,
                          "repeated_sign_errors": 1}, edits)

    def test_history_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.verify_history_hash(path, "no_stem", 1701)

    def test_initialized_candidate_must_match_all_histories(self) -> None:
        root = Path(__file__).parents[1]
        data = MODULE.load_pinned(root / "artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json",
                                  MODULE.INPUT_SHA256)
        initialized = MODULE.canonical_initialized(data, root)
        self.assertEqual(634, initialized["summary"]["target_edits"])
        with self.assertRaises(ValueError):
            MODULE.assert_matching_initializations([initialized, {"changed": True}])

    def test_generated_snapshots_cover_all_domains_and_match_repeat_summaries(self) -> None:
        root = Path(__file__).parents[1]
        data = MODULE.load_pinned(root / "artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json",
                                  MODULE.INPUT_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            MODULE._write_outputs(data, root / "artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json", Path(tmp))
            rows = [json.loads(line) for line in (Path(tmp) / "per_example_predictions.jsonl").read_text().splitlines()]
        expected = {"baseline_selector": data["baseline"]["domains"],
                    "initialized_candidate": MODULE.canonical_initialized(data, root)["domains"]}
        expected.update({f"best_{row['arm']}_{row['seed']}": row["validation"]["domains"] for row in data["results"]})
        self.assertEqual(set(expected), {row["snapshot"] for row in rows})
        self.assertEqual(6 * 1212, len(rows))
        for snapshot, domains in expected.items():
            self.assertEqual(set(domains), {row["domain"] for row in rows if row["snapshot"] == snapshot})
            for domain, summary in domains.items():
                total = sum(row["repeated_sign_errors"] for row in rows
                            if row["snapshot"] == snapshot and row["domain"] == domain)
                self.assertEqual(summary["repeated_sign_errors"], total)

    def test_pin_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_pinned(path, "0" * 64)

    def test_real_result_is_failure(self) -> None:
        root = Path(__file__).parents[1]
        data = MODULE.load_pinned(root / "artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json",
                                  MODULE.INPUT_SHA256)
        result = MODULE.select(data["results"])
        self.assertIsNone(result["selected_arm"])
        self.assertFalse(result["integration_allowed"])


if __name__ == "__main__":
    unittest.main()
