import unittest
from types import SimpleNamespace
import numpy as np
from scripts import live_stage2_ctc_v17 as live
from active.v17.extract_v17 import FrameDetection

class LiveInputTests(unittest.TestCase):
    def test_training_and_runtime_observer_preserves_sparse_phase_and_resolution(self):
        calls=[]
        def detect(frame,**kwargs):
            calls.append((frame.shape,kwargs))
            return FrameDetection([],np.ones((4,2)),np.ones(4),np.ones((15,2)),np.ones(15))
        args=SimpleNamespace(detection_image_side=640,maximum_image_side=1280,dense_model_auxiliary=False)
        detector=SimpleNamespace(detect=detect)
        frame=np.zeros((720,1280,3),np.uint8)
        wrists={'left':None,'right':None}
        a=live.observe_stage2_frame(frame,0.,0,detector,wrists,args)
        b=live.observe_stage2_frame(frame,.05,1,detector,wrists,args)
        self.assertEqual(calls[0][0],(360,640,3))
        self.assertTrue(calls[0][1]['include_body'])
        self.assertFalse(calls[1][1]['include_face'])
        self.assertEqual(a.frame.shape,frame.shape)
        self.assertTrue(a.face_for_features)
        self.assertFalse(b.face_for_features)
        self.assertFalse(b.detection.body_confidence.any())

    def test_adapted_runtime_requires_exact_live_contract_and_teacher(self):
        from scripts.cache_stage2_live_matched_v17 import input_contract
        import tempfile
        from pathlib import Path
        args=SimpleNamespace(processing_fps=20.,detection_image_side=640,maximum_image_side=1280,
            minimum_point_confidence=.15,stage2_window_seconds=32/30,sequence_preview=True,
            dense_model_auxiliary=False)
        with tempfile.TemporaryDirectory() as root:
            args.stage2_other_preservation=Path(root)/'teacher.pth'
            args.stage2_other_preservation.write_bytes(b'teacher')
            payload=dict(input_contract=input_contract(args),design=dict(teacher_sha256=live.sha256(args.stage2_other_preservation)))
            live.validate_live_adapted_runtime(args,payload)
            args.sequence_preview=False
            args.revisable_transcript=True
            live.validate_live_adapted_runtime(args,payload)
            args.detection_image_side=1280
            with self.assertRaises(ValueError):live.validate_live_adapted_runtime(args,payload)

if __name__=='__main__': unittest.main()
