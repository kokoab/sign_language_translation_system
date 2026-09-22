"""Voice selection for spoken output.  Resolution only; nothing is spoken."""

from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.live_isolated_v17 import (
    DEFAULT_SPEECH_RATE,
    PREFERRED_VOICES,
    resolve_voice,
)

COMPACT = "com.apple.voice.compact.en-US.Samantha"
PREMIUM = "com.apple.voice.premium.en-US.Ava"
ZOE = "com.apple.voice.premium.en-US.Zoe"
KAREN = "com.apple.voice.compact.en-AU.Karen"


class ResolveVoiceTest(unittest.TestCase):
    def test_the_softest_installed_voice_wins(self) -> None:
        self.assertEqual(resolve_voice(None, [COMPACT, PREMIUM, KAREN]), PREMIUM)

    def test_it_falls_back_through_the_preference_order(self) -> None:
        self.assertEqual(resolve_voice(None, [COMPACT, ZOE]), ZOE)
        self.assertEqual(resolve_voice(None, [COMPACT, KAREN]), COMPACT)

    def test_nothing_recognised_keeps_the_system_default(self) -> None:
        self.assertIsNone(resolve_voice(None, [KAREN]))

    def test_a_full_identifier_is_honoured(self) -> None:
        self.assertEqual(resolve_voice(KAREN, [COMPACT, KAREN]), KAREN)

    def test_a_short_name_is_matched_case_insensitively(self) -> None:
        self.assertEqual(resolve_voice("ava", [COMPACT, PREMIUM]), PREMIUM)
        self.assertEqual(resolve_voice("KAREN", [COMPACT, KAREN]), KAREN)

    def test_a_missing_request_falls_back_rather_than_failing(self) -> None:
        """A voice that is not installed must never stop a session."""
        self.assertEqual(resolve_voice("Ava (Premium)", [COMPACT]), COMPACT)
        self.assertEqual(resolve_voice("nonsense", [COMPACT, PREMIUM]), PREMIUM)

    def test_an_empty_request_is_treated_as_no_request(self) -> None:
        self.assertEqual(resolve_voice("", [COMPACT, PREMIUM]), PREMIUM)
        self.assertEqual(resolve_voice("   ", [COMPACT, PREMIUM]), PREMIUM)

    def test_no_voices_at_all_is_survivable(self) -> None:
        self.assertIsNone(resolve_voice(None, []))
        self.assertIsNone(resolve_voice("ava", []))


class PreferenceTest(unittest.TestCase):
    def test_premium_is_preferred_over_enhanced_over_compact(self) -> None:
        tiers = [identifier.split(".")[3] for identifier in PREFERRED_VOICES]
        self.assertEqual(tiers[0], "premium")
        self.assertEqual(tiers[-1], "compact")
        self.assertLess(tiers.index("premium"), tiers.index("enhanced"))
        self.assertLess(tiers.index("enhanced"), tiers.index("compact"))

    def test_the_last_resort_is_a_voice_macos_always_ships(self) -> None:
        self.assertEqual(PREFERRED_VOICES[-1], COMPACT)

    def test_the_default_rate_is_calmer_than_the_old_hard_coded_one(self) -> None:
        self.assertLess(DEFAULT_SPEECH_RATE, 220.0)


if __name__ == "__main__":
    unittest.main()
