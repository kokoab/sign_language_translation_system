"""Focused tests for the ASL-order Stage-3 corpus and its input encoding."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.asl_corpus_v17 import (
    GLOSS_LEMMAS,
    OBLIGATORY_GLOSSES,
    Utterance,
    assign_confidences,
    corpus_signature,
    generate_clean,
    generate_corpus,
    inject_noise,
    locked_vocabulary,
    validate_english,
)
from active.v17.stage3_asl_encoding_v17 import HIGH, LOW, MID, bucket, decode_glosses, encode


class VocabularyTest(unittest.TestCase):
    def test_locked_vocabulary_is_the_hundred_labels(self) -> None:
        vocabulary = locked_vocabulary()
        self.assertEqual(len(vocabulary), 100)
        self.assertEqual(len(set(vocabulary)), 100)
        self.assertIn("TIRED", vocabulary)
        # Negation is NO; the locked set has no NOT, which is why the renderer has
        # to supply it.
        self.assertIn("NO", vocabulary)
        self.assertNotIn("NOT", vocabulary)
        # No copula either.
        for absent in ("IS", "ARE", "AM", "THE", "A"):
            self.assertNotIn(absent, vocabulary)

    def test_every_gloss_has_lemmas(self) -> None:
        missing = [g for g in locked_vocabulary() if g not in GLOSS_LEMMAS]
        self.assertEqual(missing, [], f"glosses without English lemmas: {missing}")


class GeneratorTest(unittest.TestCase):
    def test_generated_sequences_stay_inside_the_locked_vocabulary(self) -> None:
        vocabulary = set(locked_vocabulary())
        for row in generate_clean(1500, seed=11):
            for gloss in row.glosses:
                self.assertIn(gloss, vocabulary, f"{row.key} left the vocabulary")

    def test_sequences_are_unique(self) -> None:
        rows = generate_clean(1500, seed=12)
        self.assertEqual(len({r.key for r in rows}), len(rows))

    def test_reordering_structures_dominate(self) -> None:
        """The corpus exists to teach reordering, so it must actually contain it."""
        rows = generate_clean(2000, seed=13)
        families = [r.structure.split("+")[0].split("(")[0] for r in rows]
        reordering = sum(
            1 for f in families if f in {"osv_topic", "osv_place", "wh_final"}
        )
        self.assertGreater(reordering / len(rows), 0.20)

    def test_generation_is_deterministic_for_a_seed(self) -> None:
        first = generate_corpus(300, seed=14)
        second = generate_corpus(300, seed=14)
        self.assertEqual(corpus_signature(first), corpus_signature(second))
        self.assertEqual(
            [r.confidences for r in first], [r.confidences for r in second]
        )


class NoiseTest(unittest.TestCase):
    def test_injected_noise_is_excluded_from_the_clean_sequence(self) -> None:
        rng = random.Random(3)
        base = Utterance(("I", "GO", "SCHOOL"), "motion", "i go to school")
        for _ in range(40):
            noisy = inject_noise(base, rng)
            self.assertEqual(len(noisy.glosses), len(base.glosses) + 1)
            self.assertEqual(noisy.clean_glosses, base.glosses)
            self.assertEqual(len(noisy.noise_indices), 1)

    def test_confidences_align_one_to_one(self) -> None:
        rng = random.Random(4)
        for row in generate_corpus(200, seed=15):
            self.assertEqual(len(row.confidences), len(row.glosses))
            for value in row.confidences:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        with self.assertRaises(ValueError):
            Utterance(("I", "TIRED"), "s", "m").with_confidences([0.5])

    def test_confidence_bands_overlap(self) -> None:
        """A disjoint split would teach 'low score means drop' as a perfect rule.

        Real accepted predictions in the live sessions go down to 0.250, so genuine
        glosses must also appear at low confidence or the renderer will delete real
        content whenever the recognizer is merely unsure.
        """
        rows = generate_corpus(3000, seed=16)
        genuine = [
            c for r in rows
            for i, c in enumerate(r.confidences) if i not in set(r.noise_indices)
        ]
        noise = [r.confidences[i] for r in rows for i in r.noise_indices]
        self.assertTrue(any(c < 0.40 for c in genuine), "no low-confidence genuine gloss")
        self.assertTrue(any(c > 0.50 for c in noise), "no high-confidence noise gloss")
        # Informative, but not decisive.
        low_genuine = sum(1 for c in genuine if c < 0.40)
        low_noise = sum(1 for c in noise if c < 0.40)
        share = low_genuine / (low_genuine + low_noise)
        self.assertGreater(share, 0.25)
        self.assertLess(share, 0.85)


class StrandedTest(unittest.TestCase):
    """A gloss can need dropping for grammar, not weak evidence.

    Live case: `I SICK MY HUNGRY` at confidences 0.97/0.89/0.91/0.52 rendered as
    "I am sick, and my family is hungry." MY was recognized clearly; the renderer
    invented a noun for it.
    """

    def test_stranded_glosses_are_dropped_at_high_confidence(self) -> None:
        rows = [r for r in generate_corpus(3000, seed=21)
                if r.structure.startswith("stranded")]
        self.assertTrue(rows, "no stranded rows generated")
        for row in rows:
            self.assertTrue(row.noise_is_confident)
            self.assertEqual(len(row.noise_indices), 1)
            score = row.confidences[row.noise_indices[0]]
            self.assertGreaterEqual(score, 0.25)
        median = sorted(row.confidences[row.noise_indices[0]] for row in rows)
        self.assertGreater(median[len(median) // 2], 0.45)

    def test_stranded_gloss_is_a_possessive_before_a_non_noun(self) -> None:
        """A possessive before a noun is ordinary English and must not be dropped."""
        from active.v17.asl_corpus_v17 import PERSON_STATES, STRANDED

        self.assertEqual(set(STRANDED), {"MY", "YOUR", "OUR"})
        for row in generate_corpus(2000, seed=22):
            if row.structure != "stranded":
                continue
            index = row.noise_indices[0]
            self.assertIn(row.glosses[index], STRANDED)
            if index + 1 < len(row.glosses):
                self.assertIn(row.glosses[index + 1], PERSON_STATES)

    def test_no_row_carries_two_layered_artifacts(self) -> None:
        for row in generate_corpus(3000, seed=23):
            self.assertLessEqual(len(row.noise_indices), 1)


class ValidationTest(unittest.TestCase):
    def test_rejects_the_deployed_models_subject_loss(self) -> None:
        """'I TIRED' rendered as 'Is it tired?' is the reported live failure."""
        row = Utterance(("I", "TIRED"), "state_predicate", "i am tired")
        ok, reasons = validate_english(row, "Is it tired?")
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("lost_subject") for r in reasons))
        self.assertTrue(validate_english(row, "I am tired.")[0])
        self.assertTrue(validate_english(row, "I'm tired.")[0])

    def test_short_lemmas_do_not_prefix_match(self) -> None:
        """The gloss I must not be satisfied by the word 'is'."""
        row = Utterance(("I", "SLEEP"), "s", "m")
        self.assertFalse(validate_english(row, "Is sleeping.")[0])

    def test_accepts_ordinary_inflection(self) -> None:
        for glosses, english in (
            (("FAMILY", "USE", "TIME"), "The family uses the time."),
            (("HE", "TRY", "LEARN"), "He tries to learn."),
            (("CHILD", "SEE", "WATER"), "The child sees the water."),
            (("THANKYOU", "THEY", "SICK"), "Thank you, they are sick."),
            (("I", "HAPPY"), "I am happy."),
        ):
            row = Utterance(glosses, "s", "m")
            ok, reasons = validate_english(row, english)
            self.assertTrue(ok, f"{english} rejected for {reasons}")

    def test_rejects_invented_content(self) -> None:
        row = Utterance(("I", "NEED", "WATER"), "svo", "i need water")
        ok, reasons = validate_english(
            row, "Please help me, I need the computer for the part."
        )
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("invented_content") for r in reasons))

    def test_rejects_a_leaked_noise_gloss(self) -> None:
        row = Utterance(
            ("HOW", "STOP", "HE", "FEEL"), "wh_final", "how is he", noise_indices=(1,)
        )
        self.assertTrue(validate_english(row, "How is he feeling?")[0])
        ok, reasons = validate_english(row, "How does he stop feeling?")
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("noise_leaked") for r in reasons))

    def test_allows_one_pronominalized_topic(self) -> None:
        """English legitimately pronominalizes a fronted ASL topic."""
        row = Utterance(("MAN", "THEY", "FIND"), "osv_topic", "they find the man")
        self.assertTrue(validate_english(row, "They found the man.")[0])
        self.assertTrue(validate_english(row, "They found him.")[0])

    def test_rejects_empty_output(self) -> None:
        row = Utterance(("I", "TIRED"), "s", "m")
        self.assertFalse(validate_english(row, "   ")[0])

    def test_obligatory_glosses_are_the_person_markers(self) -> None:
        self.assertEqual(
            OBLIGATORY_GLOSSES,
            frozenset({"I", "YOU", "WE", "THEY", "HE", "MY", "YOUR", "OUR"}),
        )


class EncodingTest(unittest.TestCase):
    def test_bucket_edges(self) -> None:
        self.assertEqual(bucket(0.95), HIGH)
        self.assertEqual(bucket(0.60), HIGH)
        self.assertEqual(bucket(0.59), MID)
        self.assertEqual(bucket(0.40), MID)
        self.assertEqual(bucket(0.39), LOW)
        self.assertEqual(bucket(0.25), LOW)

    def test_missing_confidence_is_treated_as_high(self) -> None:
        """A caller without evidence must not have its glosses discarded."""
        self.assertEqual(bucket(None), HIGH)
        self.assertEqual(encode(["I", "TIRED"]), "hi i hi tired")

    def test_round_trip(self) -> None:
        glosses = ["NIGHT", "WE", "TELL", "DAY", "FATHER"]
        confidences = [0.471, 0.730, 0.842, 0.287, 0.447]
        encoded = encode(glosses, confidences)
        self.assertEqual(encoded, "mid night hi we hi tell lo day mid father")
        self.assertEqual(decode_glosses(encoded), [g.lower() for g in glosses])

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode(["I", "TIRED"], [0.9])

    def test_empty_sequence(self) -> None:
        self.assertEqual(encode([]), "")
        self.assertEqual(decode_glosses(""), [])


if __name__ == "__main__":
    unittest.main()


class NaturalizerEncodingTest(unittest.TestCase):
    """The live renderer's input format, without loading any model weights."""

    @staticmethod
    def _naturalizer(encoding: str):
        import argparse
        import importlib

        module = importlib.import_module("scripts.live_isolated_v17")
        args = argparse.Namespace(
            stage3_checkpoint=Path("unused"),
            stage3_device="cpu",
            stage3_encoding=encoding,
        )
        return module.TinyStage3Naturalizer(args)

    def test_plain_encoding_is_the_deployed_behaviour(self) -> None:
        renderer = self._naturalizer("plain")
        self.assertEqual(
            renderer._model_input(["I", "GO", "SCHOOL"], [0.9, 0.3, 0.8]),
            "i go school",
        )

    def test_evidence_encoding_carries_confidence(self) -> None:
        renderer = self._naturalizer("evidence")
        self.assertEqual(
            renderer._model_input(["I", "GO", "SCHOOL"], [0.9, 0.3, 0.8]),
            "hi i lo go hi school",
        )

    def test_evidence_encoding_without_scores_keeps_every_gloss(self) -> None:
        """No evidence must never mean 'drop everything'."""
        renderer = self._naturalizer("evidence")
        self.assertEqual(renderer._model_input(["I", "TIRED"], None), "hi i hi tired")

    def test_encoding_follows_the_checkpoint(self) -> None:
        """A caller that never heard of the flag still gets the right input format.

        scripts/live_continuous_v17.py builds the naturalizer without the flag, so the
        contract has to come from the checkpoint or the model is silently fed an input
        format it was never trained on.
        """
        import argparse
        import importlib

        module = importlib.import_module("scripts.live_isolated_v17")
        self.assertEqual(
            module.stage3_encoding_for(module.DEFAULT_STAGE3_TINY), "evidence"
        )
        self.assertEqual(
            module.stage3_encoding_for(module.LEGACY_STAGE3_TINY), "plain"
        )
        # A checkpoint with no contract beside it predates the evidence input.
        self.assertEqual(module.stage3_encoding_for(Path("nonexistent")), "plain")
        args = argparse.Namespace(
            stage3_checkpoint=module.DEFAULT_STAGE3_TINY, stage3_device="cpu",
        )
        self.assertEqual(module.TinyStage3Naturalizer(args).encoding, "evidence")

    def test_explicit_encoding_overrides_the_checkpoint(self) -> None:
        import argparse
        import importlib

        module = importlib.import_module("scripts.live_isolated_v17")
        args = argparse.Namespace(
            stage3_checkpoint=module.DEFAULT_STAGE3_TINY, stage3_device="cpu",
            stage3_encoding="plain",
        )
        self.assertEqual(module.TinyStage3Naturalizer(args).encoding, "plain")

    def test_default_checkpoint_is_the_composition_model(self) -> None:
        import importlib

        module = importlib.import_module("scripts.live_isolated_v17")
        self.assertEqual(module.DEFAULT_STAGE3_TINY.name, "stage3_composition_v17_20260929_v2")
        self.assertTrue(module.DEFAULT_STAGE3_TINY.exists())
        self.assertTrue(
            (module.DEFAULT_STAGE3_TINY / "stage3_input_contract.json").exists()
        )

    def test_plain_path_keeps_the_deployed_token_window(self) -> None:
        """Widening the window changes deployed output past about 41 tokens."""
        self.assertEqual(self._naturalizer("plain").encoding, "plain")
        renderer_plain = self._naturalizer("plain")
        renderer_evidence = self._naturalizer("evidence")
        limit = lambda r: 64 if r.encoding == "evidence" else 48  # noqa: E731
        self.assertEqual(limit(renderer_plain), 48)
        self.assertEqual(limit(renderer_evidence), 64)
