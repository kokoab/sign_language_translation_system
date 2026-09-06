from pathlib import Path
import unittest

import numpy as np

from scripts import render_rigged_avatar_v17 as renderer
from scripts.render_rigged_avatar_v17 import _load_makehuman, _map_segment


class AvatarMeshTest(unittest.TestCase):
    def test_camera_height_is_the_projection_axis_for_foreshortened_hands(self):
        points = np.array([[0., 1.2, 0.], [0., 1.2, .4]])
        projected = renderer.project(points, 720, 900)
        np.testing.assert_allclose(projected[:, 1], 900 * .4)
        # A hand below the camera moves down, not up, as it approaches the lens.
        points[:, 1] = 1.1
        projected = renderer.project(points, 720, 900)
        self.assertGreater(projected[1, 1], projected[0, 1])

    def test_curled_finger_surface_does_not_flip_past_ninety_degrees(self):
        from types import SimpleNamespace
        asset = {"vertices": np.array([[.01, 1., 0.]]), "weights": {
            "finger2-2.R": (np.array([0]), np.array([1.]))}, "bones": {}}
        hands = np.zeros((2, 2, 21, 3))
        for suffix in ("L", "R"):
            asset["bones"].update({f"wrist.{suffix}": (np.zeros(3), np.array([0., .5, 0.])),
                f"finger2-1.{suffix}": (np.array([-.1, 1., 0.]), np.array([-.1, 2., 0.])),
                f"finger3-1.{suffix}": (np.array([0., 1., 0.]), np.array([0., 2., 0.])),
                f"finger5-1.{suffix}": (np.array([.1, 1., 0.]), np.array([.1, 2., 0.]))})
            for index in range(1, 5):
                asset["bones"][f"metacarpal{index}.{suffix}"] = (np.zeros(3), np.array([0., 1., 0.]))
        asset["bones"]["finger2-2.R"] = (np.array([0., 1., 0.]), np.array([0., 2., 0.]))
        hands[:, :, 5] = [-.1, 1., 0.]
        hands[:, :, 9] = [0., 1., 0.]
        hands[:, :, 17] = [.1, 1., 0.]
        hands[:, :, 6] = [0., 1., 0.]
        for frame, angle in enumerate(np.deg2rad([80., 100.])):
            hands[frame, :, 7] = [0., 1. + np.cos(angle), np.sin(angle)]
        rig = SimpleNamespace(hands=hands, shoulders=np.zeros((2, 2, 3)), elbows=np.zeros((2, 2, 3)))
        positions = [renderer.pose_makehuman(asset, rig, i)[0] for i in range(2)]
        np.testing.assert_allclose(np.asarray(positions)[:, 0], .01, atol=1e-6)

    def test_loader_keeps_every_body_surface_triangle(self):
        root = Path("artifacts/tools/makehuman_cc0")
        if not (root / "base.obj").exists():
            self.skipTest("optional MakeHuman review asset unavailable")
        group, expected = "", 0
        for line in (root / "base.obj").read_text().splitlines():
            if line.startswith("g "):
                group = line.split(maxsplit=1)[1]
            elif line.startswith("f ") and group == "body":
                expected += len(line.split()) - 3
        self.assertEqual(len(_load_makehuman(root)["faces"]), expected)

    def test_opposite_bone_direction_rotates_without_mirroring_volume(self):
        points = np.eye(3, dtype=np.float32)
        moved = _map_segment(points, np.zeros(3), np.array([0., 1., 0.]),
                             np.zeros(3), np.array([0., -1., 0.]))
        self.assertAlmostEqual(np.linalg.det(moved), 1., places=6)

    def test_hand_surface_roll_follows_palm_normal(self):
        moved = _map_segment(np.array([[0., 0., 1.]]), np.zeros(3), np.array([0., 1., 0.]),
                             np.zeros(3), np.array([0., 1., 0.]),
                             old_normal=np.array([0., 0., 1.]), new_normal=np.array([1., 0., 0.]))
        np.testing.assert_allclose(moved, [[1., 0., 0.]], atol=1e-6)

    def test_palm_targets_follow_wrist_instead_of_leaving_metacarpals_at_bind_pose(self):
        from types import SimpleNamespace
        root = Path("artifacts/tools/makehuman_cc0")
        if not root.exists():
            self.skipTest("optional asset unavailable")
        asset = _load_makehuman(root)
        hand = np.zeros((1, 2, 21, 3), np.float32)
        hand[0, :, :, 0] = np.linspace(0., .15, 21)
        hand[0, :, :, 1] = 1.4
        rig = SimpleNamespace(hands=hand, shoulders=np.ones((1, 2, 3)), elbows=np.zeros((1, 2, 3)))
        self.assertTrue(hasattr(renderer, "makehuman_targets"), "palm target mapping is missing")
        targets = renderer.makehuman_targets(asset, rig, 0)
        for suffix in ("L", "R"):
            for index, node in enumerate((5, 9, 13, 17), 1):
                head, tail = targets[f"metacarpal{index}.{suffix}"]
                np.testing.assert_allclose(tail, hand[0, 0, node])
                self.assertAlmostEqual(float(head[1]), 1.4, places=5)


if __name__ == "__main__":
    unittest.main()
