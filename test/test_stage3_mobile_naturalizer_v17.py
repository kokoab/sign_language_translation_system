import copy
import hashlib
import unittest

from active.v17.stage2_stage3_contract_v17 import load_contract, make_stage2_output
from active.v17.stage3_mobile_naturalizer_v17 import (
    MANIFEST_PATH,
    literal_render,
    load_naturalizer_manifest,
    naturalize_stage2_output,
)


class Stage3MobileNaturalizerV17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()
        cls.manifest = load_naturalizer_manifest()

    def output(self, tokens):
        return make_stage2_output(
            utterance_id="unit", token_indices=tokens, window_count=1,
            contract=self.contract,
        )

    def test_reviewed_template_and_literal_are_both_exposed(self):
        result = naturalize_stage2_output(
            self.output([15, 14, 2]), manifest=self.manifest, contract=self.contract,
        )
        self.assertEqual(result["glosses"], ["HELLO", "HOW", "YOU"])
        self.assertEqual(result["literal_english"], "Hello how you.")
        self.assertEqual(result["natural_english"], "Hello, how are you?")
        self.assertEqual(result["rendering_mode"], "reviewed_template")
        self.assertFalse(result["safe_fallback_used"])

    def test_unseen_sequence_falls_back_without_deleting_tokens(self):
        result = naturalize_stage2_output(
            self.output([92, 78, 70]), manifest=self.manifest, contract=self.contract,
        )
        self.assertEqual(result["glosses"], ["FRIEND", "NOW", "SMALL"])
        self.assertEqual(result["natural_english"], "Friend now small.")
        self.assertEqual(result["natural_english"], result["literal_english"])
        self.assertEqual(result["rendering_mode"], "literal_fallback")
        self.assertTrue(result["safe_fallback_used"])

    def test_empty_sequence_is_empty_and_safe(self):
        result = naturalize_stage2_output(
            self.output([]), manifest=self.manifest, contract=self.contract,
        )
        self.assertEqual(result["literal_english"], "")
        self.assertEqual(result["natural_english"], "")
        self.assertEqual(result["rendering_mode"], "empty")
        self.assertTrue(result["safe_fallback_used"])

    def test_all_locked_labels_have_a_deterministic_literal_rendering(self):
        for index, expected in enumerate(self.contract["vocabulary"]["labels"], 1):
            result = naturalize_stage2_output(
                self.output([index]), manifest=self.manifest, contract=self.contract,
            )
            self.assertEqual(result["glosses"], [expected])
            self.assertTrue(result["literal_english"].endswith("."))
            self.assertNotEqual(result["literal_english"], "")

    def test_manifest_hash_is_pinned_in_every_output(self):
        result = naturalize_stage2_output(
            self.output([15]), manifest=self.manifest, contract=self.contract,
        )
        self.assertEqual(
            result["naturalizer_manifest_sha256"],
            hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        )

    def test_tampered_stage2_output_fails_closed(self):
        output = self.output([15])
        for key in ("vocabulary_manifest_sha256", "recognizer_checkpoint_sha256"):
            changed = copy.deepcopy(output)
            changed[key] = "0" * 64
            with self.assertRaises(ValueError):
                naturalize_stage2_output(
                    changed, manifest=self.manifest, contract=self.contract,
                )

    def test_literal_special_case_does_not_merge_glosses(self):
        self.assertEqual(
            literal_render(["THANKYOU", "GOODBYE"], self.manifest),
            "Thank you goodbye.",
        )


if __name__ == "__main__":
    unittest.main()
