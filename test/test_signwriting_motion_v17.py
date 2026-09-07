import unittest
import numpy as np

from active.v17.avatar_rig_v17 import (parse_signwriting_signbox,bone_length_metrics,
    _fallback_rest_hand,constrain_signwriting_handshape)
from active.v17.signwriting_motion_v17 import generate_manual_rig,motion_transform,oriented_hand


class SignWritingMotionV17Test(unittest.TestCase):
    def test_hand_rotation_uses_symbol_direction_independent_of_photo_roll_and_side(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        for side in (0,1):
            up=oriented_hand('S10040',side,{'S1004':template},{})
            right=oriented_hand('S10042',side,{'S1004':template},{})
            up_axis=up[9,:2]-up[0,:2];up_axis/=np.linalg.norm(up_axis)
            right_axis=right[9,:2]-right[0,:2];right_axis/=np.linalg.norm(right_axis)
            np.testing.assert_allclose(up_axis,[0,1],atol=1e-5)
            np.testing.assert_allclose(right_axis,[1,0],atol=1e-5)

    def test_right_symbol_uses_right_hand_and_left_reflection_uses_left_hand(self):
        template=np.zeros((21,3),np.float32)
        template[9]=[0,1,0];template[5]=[.4,1,0];template[17]=[-.4,1,0]
        right=oriented_hand('S10040',1,{'S1004':template},{})
        left=oriented_hand('S10048',0,{'S1004':template},{})
        self.assertGreater(right[5,0],right[17,0])
        self.assertLess(left[5,0],left[17,0])

    def test_wall_floor_and_repeated_paths_are_distinct(self):
        wall,_,_=motion_transform('S22a02',21,False)
        floor,_,_=motion_transform('S26500',21,False)
        repeated,_,_=motion_transform('S26500',61,True)
        self.assertGreater(wall[-1,0],.05);np.testing.assert_allclose(wall[-1,1:],0,atol=1e-6)
        self.assertGreater(floor[-1,2],.05);np.testing.assert_allclose(floor[-1,:2],0,atol=1e-6)
        np.testing.assert_allclose(repeated[0],repeated[40],atol=1e-6)
        np.testing.assert_allclose(repeated[20],repeated[60],atol=1e-6)

    def test_manual_rig_uses_templates_and_keeps_rest_hand_static(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        views={'S15a2':template};bases={'S15a':template}
        parsed=parse_signwriting_signbox('M507x523S15a28494x496S26500493x477')
        phonology=dict(sign_type='OneHanded',minor_location='Neutral',repeated_movement='0')
        rig=generate_manual_rig(parsed,phonology,views,bases,frames=21)
        self.assertEqual(set(rig.hand_states[:,1]),{'symbolic'})
        self.assertEqual(set(rig.hand_states[:,0]),{'rest-uncertain'})
        self.assertLess(max(bone_length_metrics(rig).values()),1e-4)
        np.testing.assert_allclose(rig.hands[:,0],np.broadcast_to(rig.hands[0,0],rig.hands[:,0].shape),atol=1e-6)
        self.assertGreater(rig.hands[-1,1,0,2]-rig.hands[0,1,0,2],.05)

    def test_body_location_places_wrist_below_contact_and_clear_of_torso(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox('M519x514S15d01496x485')
        rig=generate_manual_rig(parsed,dict(sign_type='OneHanded',minor_location='TorsoTop'),
                                {}, {'S15d':template},frames=3)
        wrist=rig.hands[0,1,0]
        self.assertLess(abs(float(wrist[0])),.15)
        self.assertGreater(float(wrist[1]),1.10);self.assertLess(float(wrist[1]),1.18)
        self.assertGreater(float(wrist[2]),.20)

    def test_flick_opens_selected_finger_without_moving_wrist_or_changing_bones(self):
        template=constrain_signwriting_handshape(_fallback_rest_hand(1)[None],'S10000',side=1)[0]
        parsed=parse_signwriting_signbox('M543x517S10000520x470S21d00525x460')
        phonology=dict(sign_type='OneHanded',minor_location='HeadAway',repeated_movement='0')
        rig=generate_manual_rig(parsed,phonology,{}, {'S100':template},frames=21)
        np.testing.assert_allclose(rig.hands[:,1,0],np.broadcast_to(rig.hands[0,1,0],(21,3)),atol=1e-6)
        self.assertGreater(np.linalg.norm(rig.hands[-1,1,8]-rig.hands[0,1,8]),.02)
        self.assertLess(max(bone_length_metrics(rig).values()),1e-4)

    def test_finger_hinge_moves_fingertips_at_a_stationary_wrist(self):
        template=constrain_signwriting_handshape(_fallback_rest_hand(1)[None],'S15a20',side=1)[0]
        parsed=parse_signwriting_signbox('M516x513S15a20487x498S22114484x487')
        rig=generate_manual_rig(parsed,dict(sign_type='OneHanded',minor_location='Neutral',
            repeated_movement='1'),{}, {'S15a':template},frames=21)
        np.testing.assert_allclose(rig.hands[:,1,0],np.broadcast_to(rig.hands[0,1,0],(21,3)),atol=1e-6)
        self.assertGreater(np.linalg.norm(rig.hands[-1,1,8]-rig.hands[0,1,8]),.02)
        self.assertLess(max(bone_length_metrics(rig).values()),1e-4)

    def test_squeeze_and_path_motion_execute_together(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox(
            'M535x533S15030509x480S26504514x518S21600515x468')
        phonology=dict(sign_type='OneHanded',minor_location='Neutral',repeated_movement='0')
        rig=generate_manual_rig(parsed,phonology,{}, {'S150':template},frames=21)
        wrist=rig.hands[:,1,0]
        self.assertGreater(np.linalg.norm(wrist[-1]-wrist[0]),.05)
        relative=rig.hands[:,1,8]-wrist
        self.assertGreater(np.linalg.norm(relative[-1]-relative[0]),.02)

    def test_written_hand_positions_cross_contacting_arms(self):
        template=constrain_signwriting_handshape(_fallback_rest_hand(1)[None],'S20300',side=1)[0]
        parsed=parse_signwriting_signbox(
            'M526x526S20500494x509S20301473x475S20309505x476')
        phonology=dict(sign_type='SymmetricalOrAlternating',minor_location='TorsoTop',
                        contact='1',repeated_movement='0')
        rig=generate_manual_rig(parsed,phonology,{}, {'S203':template},frames=3)
        self.assertLess(rig.hands[0,0,0,0],rig.hands[0,1,0,0])

    def test_written_passive_hand_survives_one_handed_phonology(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox('M525x514S15a1a479x502S18220475x486S20600503x491')
        rig=generate_manual_rig(parsed,dict(sign_type='OneHanded',minor_location='PalmBack',contact='1'),
                                {}, {'S15a':template,'S182':template},frames=3)
        self.assertTrue(np.all(rig.hand_states=='symbolic'))

    def test_arm_contact_keeps_passive_hand_when_dictionary_reuses_one_side(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox(
            'M520x522S10050490x492S15a56493x477S20500480x478')
        rig=generate_manual_rig(parsed,dict(sign_type='OneHanded',major_location='Arm',
            minor_location='WristBack',contact='1'),{}, {'S100':template,'S15a':template},frames=3)
        self.assertTrue(np.all(rig.hand_states=='symbolic'))
        self.assertLess(float(rig.hands[:,:,0,1].max()),1.25)

    def test_rub_symbol_traces_a_body_plane_circle(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox('M513x514S15a02486x485S21100493x500')
        rig=generate_manual_rig(parsed,dict(sign_type='OneHanded',major_location='Body',
            minor_location='TorsoTop',movement='Circular',repeated_movement='1'),
            {}, {'S15a':template},frames=31)
        wrist=rig.hands[:,1,0]
        self.assertGreater(np.ptp(wrist[:,0]),.04);self.assertGreater(np.ptp(wrist[:,1]),.04)
        np.testing.assert_allclose(wrist[0],wrist[-1],atol=1e-6)

    def test_repeated_hand_contact_moves_active_hand_and_holds_passive_hand(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox('M522x525S11541498x491S11549479x498S20600489x476')
        rig=generate_manual_rig(parsed,dict(sign_type='DominanceViolation',major_location='Hand',
            minor_location='FingerRadial',movement='Straight',repeated_movement='1'),
            {}, {'S115':template},frames=31)
        self.assertGreater(np.ptp(rig.hands[:,1,0],axis=0).max(),.015)
        np.testing.assert_allclose(rig.hands[:,0,0],np.broadcast_to(rig.hands[0,0,0],(31,3)),atol=1e-6)

    def test_limb_symbol_lifts_passive_forearm_without_inventing_a_handshape(self):
        template=_fallback_rest_hand(1)-_fallback_rest_hand(1)[0]
        parsed=parse_signwriting_signbox(
            'M517x538S15a38498x479S37800503x506S22f00492x462S15a5a483x524')
        rig=generate_manual_rig(parsed,dict(sign_type='OneHanded',minor_location='Neutral'),
                                {}, {'S15a':template},frames=3)
        self.assertFalse(any(str(state).startswith('rest') for state in rig.hand_states.ravel()))
        self.assertGreater(float(rig.hands[0,0,0,1]),.95)


if __name__=='__main__':unittest.main()
