import unittest

import numpy as np


class AvatarGlossBankTest(unittest.TestCase):
    def test_avatar_mirror_swaps_sides_without_changing_lengths(self):
        from active.v17.avatar_rig_v17 import RetargetedAvatar, mirror_avatar
        xyz = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        hands = np.repeat(xyz[:, :, None], 21, axis=2)
        states = np.tile(['source-estimate', 'rest-uncertain'], (4, 1))
        rig = RetargetedAvatar(xyz, xyz+1, hands, states, np.tile([True, False], (4, 1)))
        result = mirror_avatar(rig)
        np.testing.assert_array_equal(result.hands[:, 1], rig.hands[:, 0]*[-1, 1, 1])
        np.testing.assert_array_equal(result.hand_states[:, 1], rig.hand_states[:, 0])
        np.testing.assert_array_equal(mirror_avatar(result).hands, rig.hands)
        np.testing.assert_array_equal(result.source_observed[:, 1], rig.source_observed[:, 0])

    def test_source_selection_requires_exact_frozen_identity_and_train(self):
        from scripts.build_avatar_gloss_bank_v17 import exact_training_rows
        cls = dict(class_index=0, canonical_label='I', citizen_raw_gloss='ME', citizen_asl_lex_code='B_01_068')
        good = dict(split='train', class_index='0', canonical_label='I', raw_gloss='ME', asl_lex_code='B_01_068')
        rows = [good] + [dict(good, **change) for change in
            ({'split': 'test'}, {'split': 'val'}, {'raw_gloss': 'ME2'},
             {'asl_lex_code': 'other'}, {'class_index': '1'}, {'canonical_label': 'YOU'})]
        self.assertEqual(exact_training_rows(rows, cls), [good])

    def test_one_hand_metadata_does_not_animate_detected_resting_hand(self):
        from scripts.build_avatar_gloss_bank_v17 import participating_hands
        f = np.zeros((8, 61, 5), np.float32)
        f[:, :42, 3] = 1
        f[:, 21:42, 0] = np.arange(8)[:, None] * .1
        self.assertEqual(participating_hands(f, 'OneHanded'), [False, True])
        self.assertEqual(participating_hands(f, 'SymmetricalOrAlternating'), [True, True])
        f[:, :42, 3] = 0
        with self.assertRaises(ValueError):
            participating_hands(f, 'OneHanded')


if __name__ == '__main__':
    unittest.main()
