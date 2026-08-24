import unittest

from scripts.download_rit_citizen100_candidates import (
    parse_metadata,
    participant_id,
    select_candidates,
)


HEADER = (
    "Video ID number,main entry gloss label,entry/variant gloss label,occurrence label,"
    "start frame of video clip containing the sign (relative to full videos),"
    "end frame of video clip containing the sign (relative to full videos),"
    "start frame of the sign (relative to full videos),"
    "end frame of the sign (relative to full videos),Dominant start handshape,"
    "Non-dominant start handshape,Dominant end handshape,Non-dominant end handshape,"
    "full video filename,sign type,Sign clip video filename,Class Label,"
    "start frame of video clip containing the sign (relative to sign clip),"
    "end frame of video clip containing the sign (relative to sign clip),"
    "start frame of the sign (relative to sign clip),"
    "end frame of the sign (relative to sign clip)\r"
)


def row(video_id, main, variant, occurrence, full_video, clip):
    return (
        f"{video_id},{main},{variant},{occurrence},1,2,,,A,,A,,{full_video},"
        f"Lexical Signs,{clip},{main},1,2,,\r"
    )


class RITCitizen100CandidateTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "classes": [
                {
                    "class_index": 0,
                    "canonical_label": "DOCTOR",
                    "citizen_raw_gloss": "DOCTOR1",
                    "citizen_asl_lex_code": "A_03_020",
                },
                {
                    "class_index": 1,
                    "canonical_label": "HELP",
                    "citizen_raw_gloss": "HELP",
                    "citizen_asl_lex_code": "D_01_042",
                },
            ]
        }

    def test_parse_cr_delimited_metadata_and_select_match_tiers(self):
        payload = (
            HEADER
            + row("1", "DOCTOR", "DOCTOR", "DOCTOR+", "P01_V01_new.mp4", "sign_1.mp4")
            + row("2", "HELP", "HELP", "HELP+", "P02_V01_new.mp4", "sign_2.mp4")
            + row("3", "HELP", "(rd)HELP", "(rd)HELP", "P03_V01_new.mp4", "sign_3.mp4")
        ).encode()
        rows = parse_metadata(payload)
        selected = select_candidates(self.manifest, rows)
        self.assertEqual([item["clip_filename"] for item in selected], ["sign_1.mp4", "sign_2.mp4"])
        self.assertEqual(
            [item["match_tier"] for item in selected],
            ["canonical_label_only", "pinned_raw_gloss_exact"],
        )
        self.assertTrue(all(item["training_eligible"] is False for item in selected))

    def test_participant_id_is_source_identifier(self):
        self.assertEqual(participant_id({"full video filename": "P018_V03_new.mp4"}), "P018")

    def test_duplicate_clip_fails_closed(self):
        payload = (
            HEADER
            + row("1", "HELP", "HELP", "HELP", "P01_V01_new.mp4", "same.mp4")
            + row("2", "HELP", "HELP", "HELP", "P02_V01_new.mp4", "same.mp4")
        ).encode()
        with self.assertRaisesRegex(ValueError, "selected more than once"):
            select_candidates(self.manifest, parse_metadata(payload))


if __name__ == "__main__":
    unittest.main()
