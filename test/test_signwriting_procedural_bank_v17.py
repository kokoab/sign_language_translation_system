import unittest
from active.v17.avatar_rig_v17 import parse_signwriting_signbox
from scripts.build_signwriting_procedural_bank_v17 import candidate_score


class SignWritingProceduralBankV17Test(unittest.TestCase):
    def test_candidate_score_prefers_matching_contact_location_and_repetition(self):
        plain=parse_signwriting_signbox('M507x523S15a28494x496S26500493x477')
        head=parse_signwriting_signbox('M536x518S30007482x483S15a11513x482S26500516x459S20500504x465')
        p=dict(sign_type='OneHanded',contact='1',major_location='Head',movement='Straight',
               repeated_movement='0',wrist_twist='0')
        self.assertGreater(candidate_score(head,p),candidate_score(plain,p))


if __name__=='__main__':unittest.main()
