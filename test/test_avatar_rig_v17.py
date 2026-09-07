import json
import unittest

import numpy as np

from active.v17.avatar_rig_v17 import bone_length_metrics, estimate_rest_reference, retarget_avatar


class AvatarRigV17Test(unittest.TestCase):
    def test_uncertain_rest_is_relaxed_narrow_and_mirrored(self):
        from active.v17.avatar_rig_v17 import _fallback_rest_hand, HAND_EDGES, HAND_BONE_LENGTHS
        left, right = _fallback_rest_hand(0), _fallback_rest_hand(1)
        np.testing.assert_allclose(left * [-1, 1, 1], right, atol=1e-6)
        np.testing.assert_allclose([np.linalg.norm(right[b]-right[a]) for a,b in HAND_EDGES],
                                   HAND_BONE_LENGTHS, atol=1e-6)
        self.assertLess(np.linalg.norm(right[5]-right[17]), .065)
        for first in (5, 9, 13, 17):
            bones = np.diff(right[first:first+4], axis=0)
            cosine = np.dot(bones[0], bones[1]) / np.prod(np.linalg.norm(bones[:2], axis=1))
            self.assertGreater(np.degrees(np.arccos(np.clip(cosine, -1, 1))), 15)

    def test_connected_clips_keep_cores_and_do_not_reset_to_rest(self):
        from active.v17.avatar_rig_v17 import (RetargetedAvatar, connect_avatar_clips,
            animate_signwriting_pilot, _fallback_rest_hand, _solve_elbow)
        clips = []
        for hand, movement in [('S10040', 'S26500'), ('S10620', 'S23004')]:
            hands = np.stack([np.repeat(_fallback_rest_hand(0)[None], 12, axis=0),
                animate_signwriting_pilot(_fallback_rest_hand(1), hand, movement, frames=12)], axis=1)
            shoulders = np.tile([[[.1677125, 1.285, 0], [-.1677125, 1.285, 0]]], (12, 1, 1))
            elbows = np.empty_like(shoulders)
            for i in range(12):
                for side in range(2):
                    elbows[i, side], wrist = _solve_elbow(shoulders[i, side], hands[i, side, 0], side)
                    hands[i, side] += wrist - hands[i, side, 0]
            clips.append(RetargetedAvatar(shoulders, elbows, hands, np.tile(['rest', 'symbolic'], (12, 1)),
                                         np.zeros((12, 2), bool)))
        joined = connect_avatar_clips(clips, 8)
        self.assertEqual(len(joined.hands), 32)
        np.testing.assert_array_equal(joined.hands[:12], clips[0].hands)
        np.testing.assert_array_equal(joined.hands[20:], clips[1].hands)
        self.assertTrue(np.all(joined.hands[12:20, 1, 0, 1] > 1))
        self.assertTrue(np.all(joined.hand_states[12:20, 1] == 'transition'))
        self.assertFalse(joined.source_observed[12:20].any())
        self.assertLess(max(bone_length_metrics(joined).values()), 1e-5)
        for invalid in ([],):
            with self.assertRaises(ValueError):
                connect_avatar_clips(invalid, 8)
        with self.assertRaises(ValueError):
            connect_avatar_clips(clips, 0)

    def test_missing_active_wrists_do_not_pull_motion_toward_coordinate_origin(self):
        xyz = np.zeros((7, 61, 3), np.float32)
        xyz[:, 21:42] = [-.8, .7, .1]
        presence = np.zeros((7, 61), bool)
        presence[:, 21:42] = True
        timeline = {'timeline': [dict(kind='gloss', start=0, stop=7,
                                     hand_participation=[False, True])]}
        complete = retarget_avatar(xyz, presence, presence, timeline)
        xyz[[0, 1, 4, 6], 21:42] = 0
        presence[[0, 1, 4, 6], 21:42] = False
        missing = retarget_avatar(xyz, presence, presence, timeline)
        np.testing.assert_allclose(missing.hands[:, 1, 0], complete.hands[:, 1, 0], atol=1e-6)
        self.assertEqual(missing.hand_states[0, 1], 'imputed-active')
        absent = retarget_avatar(np.zeros_like(xyz), np.zeros_like(presence), np.zeros_like(presence), timeline)
        self.assertTrue(np.isfinite(absent.hands).all())
        self.assertLess(absent.hands[0, 1, 0, 1], .9)

    def test_isolated_approach_preserves_core_and_metric_anatomy(self):
        from active.v17.avatar_rig_v17 import (RetargetedAvatar, prepend_isolated_approach,
            append_isolated_release, animate_signwriting_pilot, _fallback_rest_hand, _solve_elbow)
        hands = np.stack([np.repeat(_fallback_rest_hand(0)[None], 12, axis=0),
            animate_signwriting_pilot(_fallback_rest_hand(1), 'S10040', 'S26500', frames=12)], axis=1)
        shoulders = np.tile([[[.1677125, 1.285, 0], [-.1677125, 1.285, 0]]], (12, 1, 1))
        elbows = np.empty_like(shoulders)
        for i in range(12):
            for side in range(2):
                elbows[i, side], wrist = _solve_elbow(shoulders[i, side], hands[i, side, 0], side)
                hands[i, side] += wrist - hands[i, side, 0]
        core = RetargetedAvatar(shoulders, elbows, hands, np.tile(['rest', 'observed'], (12, 1)),
                               np.tile([False, True], (12, 1)))
        result = prepend_isolated_approach(core, 16)
        for field in ('hands', 'elbows', 'shoulders', 'hand_states', 'source_observed'):
            np.testing.assert_array_equal(getattr(result, field)[16:], getattr(core, field))
        np.testing.assert_allclose(result.hands[:16, 0], np.repeat(core.hands[:1, 0], 16, axis=0))
        self.assertGreater(np.linalg.norm(result.hands[0, 1, 0] - core.hands[0, 1, 0]), .2)
        self.assertLess(np.max(np.linalg.norm(np.diff(result.hands[:17, 1, 0], axis=0), axis=1)), .06)
        self.assertFalse(result.source_observed[:16].any())
        self.assertTrue(np.isfinite(result.hands).all())
        self.assertLess(max(bone_length_metrics(result).values()), 1e-5)
        released = append_isolated_release(result, 16)
        np.testing.assert_array_equal(released.hands[:-16], result.hands)
        np.testing.assert_allclose(released.hands[-1], result.hands[0], atol=1e-6)
        self.assertTrue(np.all(np.diff(released.hands[-17:, 1, 0, 1]) <= 1e-6))
        self.assertEqual(released.hand_states[-1, 1], 'release-assumed')
        self.assertLess(max(bone_length_metrics(released).values()), 1e-5)
        self.assertIs(prepend_isolated_approach(core, 0), core)
        with self.assertRaises(ValueError):
            prepend_isolated_approach(core, -1)

    def test_approach_rotates_palm_without_fanning_its_metacarpals(self):
        from active.v17.avatar_rig_v17 import (RetargetedAvatar, prepend_isolated_approach,
            interpolate_world_hand, _fallback_rest_hand, _solve_elbow)
        from scipy.spatial.transform import Rotation
        rest = _fallback_rest_hand(1)
        rotated = (rest - rest[0]) @ Rotation.from_euler('x', 160, degrees=True).as_matrix().T + [-.2, 1.1, .28]
        hands = np.stack([_fallback_rest_hand(0), rotated])[None]
        shoulders = np.array([[[.1677125, 1.285, 0], [-.1677125, 1.285, 0]]])
        elbows = np.empty_like(shoulders)
        for side in range(2):
            elbows[0, side], wrist = _solve_elbow(shoulders[0, side], hands[0, side, 0], side)
            hands[0, side] += wrist - hands[0, side, 0]
        rig = RetargetedAvatar(shoulders, elbows, hands, np.array([['rest', 'observed']]), np.array([[False, True]]))
        h = prepend_isolated_approach(rig, 20).hands[:, 1]
        for first, last in ((5, 9), (5, 17), (9, 13), (13, 17)):
            np.testing.assert_allclose(np.linalg.norm(h[:, first] - h[:, last], axis=1),
                                       np.linalg.norm(rest[first] - rest[last]), atol=1e-6)
            middle = interpolate_world_hand(rest, rotated, .5)
            self.assertAlmostEqual(float(np.linalg.norm(middle[first] - middle[last])),
                                   float(np.linalg.norm(rest[first] - rest[last])), places=6)

    def test_sigml_bridge_preserves_sequence_and_rejects_wrong_notation(self):
        from xml.etree import ElementTree as ET
        from active.v17.avatar_rig_v17 import signwriting_pilot_sigml
        you = dict(gloss='YOU', fsw='AS10040S26500M509x523S10040494x493S26500492x477')
        need = dict(gloss='NEED', fsw='AS10620S23004M514x524S10620486x476S23004489x506')
        root = ET.fromstring(signwriting_pilot_sigml([need, you, need]))
        self.assertEqual([s.attrib['gloss'] for s in root], ['NEED', 'YOU', 'NEED'])
        self.assertEqual(len(root.findall('.//rpt_motion')), 2)
        # The rejected hooked preset also bent the fingertip joint.
        shape = root[0].find('sign_manual/handconfig')
        self.assertNotIn('mainbend', shape.attrib)
        self.assertEqual(shape.attrib['bend2'].split()[2], '0')
        for invalid in ([], [dict(gloss='HELLO', fsw=you['fsw'])],
                        [dict(gloss='YOU', fsw=need['fsw'])],
                        [dict(gloss='NEED', fsw=need['fsw'].replace('S23004', 'S22e04'))],
                        [dict(gloss='YOU', fsw=None)]):
            with self.assertRaises(ValueError):
                signwriting_pilot_sigml(invalid)

    def test_signwriting_candidate_must_match_locked_repetition(self):
        from active.v17.avatar_rig_v17 import signwriting_pilot_symbols
        single='AS10620S22e04M511x524S10620488x476S22e04494x506'
        double='AS10620S23004M514x524S10620486x476S23004489x506'
        self.assertEqual(signwriting_pilot_symbols(double,repeated=True),('S10620','S23004'))
        with self.assertRaises(ValueError):signwriting_pilot_symbols(single,repeated=True)
        with self.assertRaises(ValueError):signwriting_pilot_symbols(double,repeated=False)

    def test_signwriting_fist_keeps_spatial_symbols_and_executes_repeated_flex(self):
        from active.v17.avatar_rig_v17 import (parse_signwriting_signbox,
            signwriting_pilot_symbols, animate_signwriting_pilot, _fallback_rest_hand,
            HAND_EDGES, HAND_BONE_LENGTHS)
        fsw = 'AS20320S23004M513x518S20320493x481S23004488x500'
        parsed = parse_signwriting_signbox(fsw)
        self.assertEqual(parsed['sort_symbols'], ['S20320', 'S23004'])
        self.assertEqual(parsed['symbols'][0], dict(key='S20320', base='S203',
            fill=2, rotation=0, x=493, y=481, category='hand'))
        self.assertEqual(parsed['symbols'][1]['category'], 'movement')
        self.assertEqual(signwriting_pilot_symbols(fsw, repeated=True), ('S20320', 'S23004'))
        with self.assertRaises(ValueError):
            signwriting_pilot_symbols(fsw, repeated=False)
        for invalid in (fsw + 'junk', fsw.replace('S20320', 'S3ff20'), None):
            with self.assertRaises(ValueError): parse_signwriting_signbox(invalid)
        hand = _fallback_rest_hand(1)
        poses = animate_signwriting_pilot(hand, 'S20320', 'S23004', frames=61)
        lengths = np.array([np.linalg.norm(poses[:, b]-poses[:, a], axis=-1) for a,b in HAND_EDGES])
        np.testing.assert_allclose(lengths, np.broadcast_to(HAND_BONE_LENGTHS[:,None], lengths.shape), atol=1e-6)
        np.testing.assert_allclose(poses[0], poses[40], atol=1e-6)
        np.testing.assert_allclose(poses[20], poses[60], atol=1e-6)
        np.testing.assert_allclose(poses[:,0], np.broadcast_to(poses[0,0], (61,3)), atol=1e-6)
        for first in (5,9,13,17):
            self.assertLess(poses[0,first+3,1], poses[0,first+1,1])
        self.assertGreater(np.linalg.norm(poses[20,9]-poses[0,9]), .02)

    def test_signwriting_double_flex_executes_two_strokes_with_a_return(self):
        from active.v17.avatar_rig_v17 import animate_signwriting_pilot, _fallback_rest_hand
        poses=animate_signwriting_pilot(_fallback_rest_hand(1),'S10620','S23004',frames=61)
        np.testing.assert_allclose(poses[0],poses[40],atol=1e-6)
        np.testing.assert_allclose(poses[20],poses[60],atol=1e-6)
        self.assertGreater(np.linalg.norm(poses[20,8]-poses[0,8]),.02)

    def test_signwriting_open_hands_and_wrist_wave_preserve_shape_and_anatomy(self):
        from active.v17.avatar_rig_v17 import (constrain_signwriting_handshape,
            animate_signwriting_pilot, signwriting_pilot_symbols, _fallback_rest_hand,
            HAND_EDGES, HAND_BONE_LENGTHS)
        hand = _fallback_rest_hand(1)[None]
        flat = constrain_signwriting_handshape(hand, 'S15a20', side=1)[0]
        spread = constrain_signwriting_handshape(hand, 'S14c20', side=1)[0]
        for pose in (flat, spread):
            for start in (5,9,13,17):
                bones = np.diff(pose[start:start+4], axis=0)
                unit = bones / np.linalg.norm(bones,axis=-1,keepdims=True)
                np.testing.assert_allclose(unit, np.broadcast_to(unit[0],unit.shape), atol=1e-5)
            np.testing.assert_allclose([np.linalg.norm(pose[b]-pose[a]) for a,b in HAND_EDGES],
                                       HAND_BONE_LENGTHS, atol=1e-6)
        self.assertGreater(np.linalg.norm(spread[8]-spread[20]),np.linalg.norm(flat[8]-flat[20])+.02)
        left = constrain_signwriting_handshape(_fallback_rest_hand(0)[None], 'S15a20', side=0)[0]
        np.testing.assert_allclose(left * [-1,1,1], flat, atol=1e-6)
        fsw = 'M524x520S27206505x480S14c20477x483'
        self.assertEqual(signwriting_pilot_symbols(fsw,repeated=True), ('S14c20','S27206'))
        with self.assertRaises(ValueError): signwriting_pilot_symbols(fsw,repeated=False)
        wave = animate_signwriting_pilot(hand[0], 'S14c20', 'S27206', frames=61,wrist_flex_degrees=20)
        np.testing.assert_allclose(wave[0],wave[40],atol=1e-6)
        np.testing.assert_allclose(wave[20],wave[60],atol=1e-6)
        np.testing.assert_allclose(wave[:,0],np.broadcast_to(wave[0,0],(61,3)),atol=1e-6)
        self.assertGreater(abs(float(wave[20,12,0]-wave[0,12,0])), .04)
        push = animate_signwriting_pilot(hand[0], 'S15a28', 'S26500',frames=12)
        np.testing.assert_allclose(push[-1]-push[0],np.broadcast_to([0,0,.06],(21,3)),atol=1e-6)

    def test_signwriting_hello_keeps_head_contact_but_executes_manual_salute(self):
        from active.v17.avatar_rig_v17 import (parse_signwriting_signbox,
            signwriting_pilot_symbols, animate_signwriting_pilot, _fallback_rest_hand)
        fsw = 'M536x518S30007482x483S15a11513x482S26500516x459S20500504x465'
        parsed = parse_signwriting_signbox(fsw)
        self.assertEqual([s['category'] for s in parsed['symbols']],
                         ['head_face','hand','movement','contact'])
        self.assertEqual(signwriting_pilot_symbols(fsw,repeated=False),('S15a11','S26500'))
        invalid = fsw.replace('S20500','S2f700')
        with self.assertRaises(ValueError): signwriting_pilot_symbols(invalid,repeated=False)
        poses = animate_signwriting_pilot(_fallback_rest_hand(1),'S15a11','S26500',frames=12,
            wrist_position=(-.16,1.38,.16),travel_metres=.08)
        np.testing.assert_allclose(poses[0,0],[-.16,1.38,.16],atol=1e-6)
        np.testing.assert_allclose(poses[-1,0],[-.16,1.38,.24],atol=1e-6)

    def test_symbol_motion_points_forward_and_flexes_need_at_stationary_wrist(self):
        from active.v17.avatar_rig_v17 import animate_signwriting_pilot, _fallback_rest_hand
        hand = _fallback_rest_hand(1)
        you = animate_signwriting_pilot(hand, 'S10040', 'S26500', frames=12)
        need = animate_signwriting_pilot(hand, 'S10620', 'S22e04', frames=12)
        np.testing.assert_allclose(you[-1,0]-you[0,0],[0,0,.06],atol=1e-6)
        axis=you[0,8]-you[0,5]
        self.assertLess(np.linalg.norm(axis[:2]),1e-6)
        self.assertGreater(axis[2],0)
        np.testing.assert_allclose(need[:,0],np.repeat(need[:1,0],12,axis=0))
        self.assertGreater(np.linalg.norm(need[-1,8]-need[0,8]),.02)
        calibrated=animate_signwriting_pilot(hand,'S10040','S26500',frames=12,
                                             wrist_position=(-.10,1.20,.28))
        np.testing.assert_allclose(calibrated-you,np.broadcast_to([.08,.10,0],you.shape),atol=1e-6)
        with self.assertRaises(ValueError):animate_signwriting_pilot(hand,'S10041','S26500')

    def test_signwriting_fingers_curl_toward_palm_not_back_of_hand(self):
        from active.v17.avatar_rig_v17 import constrain_signwriting_handshape, _fallback_rest_hand
        hand = _fallback_rest_hand(1)[None]
        # Right palm facing +Z: index is on +X side of the pinky, fingers +Y.
        hand[0,0] = [0,0,0]
        for node,x in [(5,.03),(9,.01),(13,-.01),(17,-.03)]:
            for j in range(4):hand[0,node+j] = [x,.08+j*.02,0]
        result = constrain_signwriting_handshape(hand,'S10620',side=1)
        self.assertGreater(result[0,7,2], result[0,6,2])
        hand[0,6] = hand[0,5] + [0,0,.02]
        with self.assertRaises(ValueError):
            constrain_signwriting_handshape(hand,'S10620',side=1)

    def test_signwriting_index_and_bent_index_keep_distinct_fixed_length_shapes(self):
        from active.v17.avatar_rig_v17 import constrain_signwriting_handshape, _fallback_rest_hand, HAND_EDGES
        hand = _fallback_rest_hand(1)[None]
        straight = constrain_signwriting_handshape(hand, "S10040", side=1)
        bent = constrain_signwriting_handshape(hand, "S10620", side=1)
        for pose in (straight, bent):
            for parent, child in HAND_EDGES:
                self.assertAlmostEqual(float(np.linalg.norm(pose[0,child]-pose[0,parent])),
                                       float(np.linalg.norm(hand[0,child]-hand[0,parent])), places=6)
        def bend(pose):
            bones = np.diff(pose[0,5:9],axis=0)
            return float(np.dot(bones[0],bones[1]) / np.prod(np.linalg.norm(bones[:2],axis=1)))
        self.assertGreater(bend(straight), .999)
        self.assertLess(abs(bend(bent)), .01)
        np.testing.assert_allclose(straight[:,9:],bent[:,9:])
        with self.assertRaises(ValueError):
            constrain_signwriting_handshape(hand, "S18011", side=1)

    def test_fully_open_annotation_extends_only_selected_fingers(self):
        from active.v17.avatar_rig_v17 import constrain_annotated_handshape
        hand = np.zeros((1, 21, 3), np.float32)
        hand[0, 5:9] = [[0, 0, 0], [.03, 0, 0], [.03, .02, 0], [.01, .02, 0]]
        hand[0, 9:13] = [[0, 0, 0], [0, .03, 0], [.02, .03, 0], [.02, .01, 0]]
        corrected = constrain_annotated_handshape(hand, "i", "FullyOpen")
        bones = np.diff(corrected[0, 5:9], axis=0)
        directions = bones / np.linalg.norm(bones, axis=1, keepdims=True)
        np.testing.assert_allclose(directions, np.repeat(directions[:1], 3, axis=0), atol=1e-6)
        np.testing.assert_array_equal(corrected[:, 9:13], hand[:, 9:13])
        np.testing.assert_array_equal(constrain_annotated_handshape(hand, "i", "Bent"), hand)

    def test_body_relative_coordinates_use_avatar_shoulder_width_isotropically(self):
        from active.v17.avatar_rig_v17 import _source_to_world
        points = _source_to_world(np.array([[.5, 0, 0], [-.5, 0, 0], [0, .5, 0], [0, 0, 0]], np.float32))
        self.assertAlmostEqual(float(points[0, 0] - points[1, 0]), .335425, places=6)
        self.assertAlmostEqual(float(points[0, 0] - points[3, 0]), float(points[3, 1] - points[2, 1]), places=6)

    def test_low_wrist_does_not_raise_elbow_above_shoulder(self):
        from active.v17.avatar_rig_v17 import _solve_elbow
        shoulder = np.array([-.1677, 1.285, 0.], np.float32)
        elbow, _ = _solve_elbow(shoulder, np.array([-.4, 1.1, .235], np.float32), 1)
        self.assertLess(float(elbow[1]), float(shoulder[1]))

    def test_right_hand_on_viewer_left_is_not_mirrored(self):
        xyz = np.zeros((5, 61, 3), np.float32)
        xyz[:, 21:42, 0] = -.4
        xyz[:, 21:42, 1] = .2
        observed = np.zeros((5, 61), bool); observed[:, 21:42] = True
        rig = retarget_avatar(xyz, observed, observed, {})
        self.assertTrue(np.all(rig.hands[:, 1, 0, 0] < 0))
        self.assertTrue(np.all(rig.shoulders[:, 1, 0] < 0))

    def test_fixed_anatomy_and_explicit_rest(self):
        frames = 12
        xyz = np.zeros((frames, 61, 3), dtype=np.float32)
        presence = np.zeros((frames, 61), dtype=bool)
        observations = np.zeros_like(presence)
        for frame in range(frames):
            scale = 1 + frame * .15
            xyz[frame, 21:42, 0] = np.linspace(-.2, .4, 21) * scale
            xyz[frame, 21:42, 1] = np.linspace(.1, .7, 21) * scale
            presence[frame, 21:42] = True
            observations[frame, 21:42] = frame % 3 != 0
        metadata = {"timeline": [{"kind": "gloss", "gloss": "HELLO", "start": 0,
                                  "stop": frames, "hand_participation": [False, True]}]}
        rig = retarget_avatar(xyz, presence, observations, json.dumps(metadata))
        self.assertTrue(np.all(rig.hand_states[:, 0] == "rest-uncertain"))
        self.assertIn("imputed-active", rig.hand_states[:, 1])
        self.assertLess(float(rig.hands[:, 0, 0, 1].mean()), .8)  # beside hip, not stomach
        for value in bone_length_metrics(rig).values():
            self.assertLess(value, 2e-5)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            retarget_avatar(np.zeros((2, 42, 3)), np.zeros((2, 42)), np.zeros((2, 42)), {})

    def test_world_hand_depth_preserves_foreshortening(self):
        xyz = np.zeros((5, 61, 3), np.float32)
        xyz[:, 21:42, 0] = -.4
        xyz[:, 21:42, 1] = .2
        observed = np.zeros((5, 61), bool); observed[:, 21:42] = True
        world = np.full((5, 2, 21, 3), np.nan, np.float32)
        world[:, 1] = 0
        world[:, 1, :, 2] = -np.arange(21) * .01
        rig = retarget_avatar(xyz, observed, observed, {}, hand_world_xyz=world)
        # A finger pointing at the camera stays short in the projected XY plane.
        delta = rig.hands[:, 1, 8] - rig.hands[:, 1, 5]
        self.assertLess(float(np.linalg.norm(delta[:, :2], axis=-1).max()), 1e-5)
        self.assertGreater(float(delta[:, 2].min()), .08)

    def test_world_transition_rotates_without_bone_collapse(self):
        from active.v17.avatar_rig_v17 import interpolate_world_hand, HAND_EDGES
        left = np.zeros((21, 3), np.float32)
        for parent, child in HAND_EDGES:
            left[child] = left[parent] + [0., .02, .01]
        right = -left
        middle = interpolate_world_hand(left, right, .5)
        for parent, child in HAND_EDGES:
            self.assertAlmostEqual(float(np.linalg.norm(middle[child] - middle[parent])), np.sqrt(.0005), places=6)
        np.testing.assert_allclose(interpolate_world_hand(left, right, 0), left, atol=1e-7)
        np.testing.assert_allclose(interpolate_world_hand(left, right, 1), right, atol=1e-7)

    def test_observed_hands_clear_front_of_torso_with_fixed_anatomy(self):
        xyz = np.zeros((8, 61, 3), np.float32)
        xyz[:, :42, 1] = .45
        xyz[:, :42, 2] = -10  # extreme scale proxy cannot send the hands into the body
        presence = np.ones((8, 61), bool)
        rig = retarget_avatar(xyz, presence, presence, {})
        self.assertGreater(float(rig.hands[..., 2].min()), .18)
        for value in bone_length_metrics(rig).values():
            self.assertLess(value, 2e-5)

    def test_rest_reference_comes_from_observed_frames(self):
        values = np.zeros((8, 61, 5), dtype=np.float32)
        values[:, :42, 3:] = 1
        values[:, 0, 0] = .35
        values[:, 0, 1] = .61
        values[:, 21, 0] = np.arange(8) * .1
        values[:, 21, 1] = .4
        reference = estimate_rest_reference([("train/P51/example.npz", values)])
        self.assertGreater(reference.candidate_counts[0], 0)
        self.assertEqual(reference.source_paths, ("train/P51/example.npz",))


if __name__ == "__main__":
    unittest.main()
