import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from active.v17.stage2_stage3_contract_v17 import (
    collapse_ctc_tokens,
    load_contract,
    make_stage2_output,
    validate_stage2_output,
)
from active.v17.export_stage1_coreml_v17 import tree_sha256


class Stage2Stage3ContractV17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()

    def test_contract_pins_locked_vocabulary_and_models(self):
        labels = self.contract["vocabulary"]["labels"]
        self.assertEqual(len(labels), 100)
        self.assertEqual(len(set(labels)), 100)
        vocabulary = Path(self.contract["vocabulary"]["manifest"])
        self.assertEqual(
            hashlib.sha256(vocabulary.read_bytes()).hexdigest(),
            self.contract["vocabulary"]["manifest_sha256"],
        )
        checkpoint = Path(self.contract["recognizer"]["checkpoint"])
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            self.contract["recognizer"]["checkpoint_sha256"],
        )
        for payload in self.contract["recognizer"]["coreml_packages"].values():
            self.assertEqual(tree_sha256(Path(payload["path"])), payload["tree_sha256"])

    def test_ctc_collapse_and_one_based_mapping(self):
        logits = np.full((64, 101), -10.0, dtype=np.float32)
        raw = [0, 15, 15, 0, 58, 58, 0, 81]
        for time, token in enumerate(raw):
            logits[time, token] = 10.0
        tokens = collapse_ctc_tokens(logits, window_count=1, contract=self.contract)
        self.assertEqual(tokens, [15, 58, 81])
        output = make_stage2_output(
            utterance_id="example", token_indices=tokens, window_count=1,
            contract=self.contract,
        )
        self.assertEqual(output["glosses"], ["HELLO", "GOOD", "MORNING"])
        validate_stage2_output(output, contract=self.contract)

    def test_empty_sequence_is_valid_and_oov_fails_closed(self):
        empty = make_stage2_output(
            utterance_id="empty", token_indices=[], window_count=1,
            contract=self.contract,
        )
        self.assertEqual(empty["glosses"], [])
        with self.assertRaises(ValueError):
            make_stage2_output(
                utterance_id="bad", token_indices=[101], window_count=1,
                contract=self.contract,
            )

    def test_hash_or_label_tampering_fails(self):
        output = make_stage2_output(
            utterance_id="hello", token_indices=[15], window_count=1,
            contract=self.contract,
        )
        for key, value in (
            ("vocabulary_manifest_sha256", "0" * 64),
            ("recognizer_checkpoint_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(output)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_stage2_output(changed, contract=self.contract)
        changed = copy.deepcopy(output)
        changed["glosses"] = ["GOODBYE"]
        with self.assertRaises(ValueError):
            validate_stage2_output(changed, contract=self.contract)


if __name__ == "__main__":
    unittest.main()
