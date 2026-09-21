import unittest
from scripts.salvage_verified_phrases_v17 import trusted_runs

class VerifiedSpanTests(unittest.TestCase):
    def test_no_gaps_ambiguous_events_overlap_or_cut_edges(self):
        def event(start,end,**kw):
            return dict(start_frame=start,end_frame_exclusive=end,status='known',complete_event=True,overlapping_event=False,**kw)
        a,b=event(0,8),event(8,16)
        self.assertEqual(trusted_runs([b,a]),[[a,b]])
        for key,value in [('status','unresolved'),('complete_event',False),('overlapping_event',True),('start_frame',9)]:
            self.assertEqual(trusted_runs([a,{**b,key:value}]),[])
        self.assertEqual(trusted_runs([{**a,'status':'oov'},{**b,'status':'oov'}]),[])
        self.assertEqual(trusted_runs([a,{**b,'status':'oov'}]),[[a,{**b,'status':'oov'}]])
