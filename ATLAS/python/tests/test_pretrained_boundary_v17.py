import unittest
import numpy as np

class PretrainedBoundaryTest(unittest.TestCase):
    def test_window_clock_future_limit_and_tail(self):
        from active.v17.pretrained_boundary_v17 import window_indices
        times=np.arange(100)/20
        idx,valid=window_indices(30,len(times))
        self.assertEqual(idx[53],30)
        self.assertEqual(idx[valid][-1],40)
        self.assertEqual(int(valid.sum()),41)
        idx,valid=window_indices(98,len(times))
        self.assertEqual(idx[53],98)
        self.assertEqual(idx[valid][-1],99)
        self.assertFalse(valid[-1])

    def test_no_early_convergence_stop(self):
        from active.v17.pretrained_boundary_v17 import should_stop
        self.assertFalse(should_stop(39,1))
        self.assertTrue(should_stop(40,1))
        self.assertFalse(should_stop(60,50))
        self.assertTrue(should_stop(120,119))

    def test_future_changes_do_not_change_window_normalization(self):
        from active.v17.pretrained_boundary_v17 import load_pose,window_features
        from pathlib import Path
        import json
        manifest=json.loads(Path('artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest.json').read_text())
        row=next(r for r in manifest['records'] if r['pose_frames']>120)
        pose=load_pose(row)
        x,t=window_features(pose,10)
        modified=load_pose(row)
        modified.body.data[21:]+=10000
        y,u=window_features(modified,10)
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(t,u)
        self.assertEqual(x.shape,(64,50,6))
        self.assertTrue(np.isfinite(x).all())

if __name__=='__main__':unittest.main()

class PretrainedCacheTest(unittest.TestCase):
    def test_cached_path_matches_raw_and_attention_adapts(self):
        import torch
        from active.v17.pretrained_boundary_v17 import PretrainedBoundary,FRAMES,TARGET,FPS
        model=PretrainedBoundary().eval()
        x=torch.randn(2,FRAMES,50,6)
        times=(torch.arange(FRAMES)-TARGET)[None]/FPS
        with torch.no_grad():
            direct=model(x,times)
            cached=model.forward_projected(model.project(x),times)
        torch.testing.assert_close(direct,cached)
        model.train();model.set_adaptation(True)
        self.assertFalse(model.backbone.frame_cnn.training)
        self.assertTrue(model.backbone.encoder_attn[0].training)
        self.assertTrue(model.backbone.encoder_attn[-1].training)
        self.assertTrue(all(p.requires_grad for p in model.backbone.encoder_attn[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.encoder_attn[-1].parameters()))

class AugmentedBoundaryTest(unittest.TestCase):
    def test_observation_jitter_is_past_only_and_preserves_padding(self):
        from active.v17.pretrained_boundary_v17 import observation_indices,window_indices
        idx,valid=window_indices(10,100)
        selected=observation_indices(valid,np.random.default_rng(123),sensor_fps=15,dropout=.15)
        self.assertTrue(np.all(selected[valid]<=np.flatnonzero(valid)))
        self.assertTrue(np.all(valid[selected[valid]]))
        self.assertTrue(np.any(selected[valid]!=np.flatnonzero(valid)))
        clean=observation_indices(valid,np.random.default_rng(123),sensor_fps=20,dropout=0)
        np.testing.assert_array_equal(clean,np.arange(64))
        with self.assertRaises(ValueError):observation_indices(valid,np.random.default_rng(1),sensor_fps=0)

    def test_augmented_features_deterministic_and_future_bounded(self):
        import json
        from pathlib import Path
        from active.v17.pretrained_boundary_v17 import load_pose,window_features
        rows=json.loads(Path('artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest.json').read_text())['records']
        row=next(r for r in rows if r['pose_frames']>120)
        pose=load_pose(row);kw=dict(augmentation_seed=17)
        a,t=window_features(pose,10,**kw)
        pose.body.data[21:]+=10000
        b,u=window_features(pose,10,**kw)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(t,u)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(a[:43]==0))

    def test_augmented_stop_has_no_arbitrary_floor(self):
        from scripts.train_pretrained_boundary_v17 import converged
        recipe=dict(maximum_epochs=80,minimum_epochs=0,patience=8,warmup_epochs=5)
        self.assertFalse(converged(7,1,recipe))
        self.assertFalse(converged(12,1,recipe))
        self.assertTrue(converged(15,7,recipe))
        self.assertFalse(converged(40,39,recipe))
        self.assertTrue(converged(80,79,recipe))

class AugmentedCacheAlignmentTest(unittest.TestCase):
    def test_variants_keep_row_targets_and_clean_evaluation(self):
        from scripts.train_pretrained_boundary_v17 import cached_batch
        x=np.arange(20,dtype=np.float32).reshape(10,2)
        y=np.arange(10,dtype=np.float32)[:,None]
        ids=np.array([7,1,9,0]);variants=[x,x+100,x+200]
        clean,labels=cached_batch(x,y,ids)
        np.testing.assert_array_equal(clean,x[ids]);np.testing.assert_array_equal(labels,y[ids])
        np.random.seed(4)
        augmented,labels=cached_batch(x,y,ids,variants)
        np.testing.assert_array_equal(labels,y[ids])
        delta=augmented-x[ids]
        self.assertTrue(np.all(np.isin(delta,[0,100,200])))
        self.assertTrue(np.all(delta[:,0]==delta[:,1]))
        self.assertTrue(np.any(delta))
        np.testing.assert_array_equal(x,np.arange(20).reshape(10,2))

class EvaluationDestinationTest(unittest.TestCase):
    def test_membership_uses_requested_report(self):
        import json,tempfile
        from pathlib import Path
        from scripts.evaluate_temporal_boundary_v17 import evaluation_rows
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)
            rows=evaluation_rows(report=destination)
            saved=json.loads((destination/'evaluation_membership.json').read_text())
            self.assertEqual(len(rows),12)
            self.assertEqual(len(saved['videos']),12)
            self.assertEqual(len(saved['inputs']),2)

class OriginalBioReadoutTest(unittest.TestCase):
    def test_adapted_checkpoint_can_use_original_bio(self):
        import torch
        from scripts import evaluate_boundary_expanded_v17 as e
        model=e.PretrainedBoundary().to('mps').eval()
        spec=e.ARMS[1]
        state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        z=torch.zeros(2,e.FRAMES,384)
        times=torch.tensor((np.arange(e.FRAMES)-e.TARGET)[None]/e.FPS,dtype=torch.float32,device='mps')
        with torch.inference_mode():
            expected=model.backbone.sign_bio_head(model.encode_projected(z.to('mps'),times)).log_softmax(-1).cpu()
        seen=[]
        def decode(values):
            torch.testing.assert_close(values,expected)
            seen.append(True)
            return []
        row=dict(source_item_id='readout-check',events=[],all_events=[],subset='check',target_sequence=['WHY'])
        cache={'readout-check':dict(projected=z,observations=[],projection_seconds=0,pose_sha256='synthetic')}
        _,records,_,_=e.run_arm(spec,[row],cache,model,None,None,times,
            dict(filter_segments=lambda x:x,likeliest_probs_to_segments=decode),readout='bio')
        self.assertEqual(seen,[True])
        self.assertEqual(records[0]['hypothesis'],[])
        with self.assertRaises(ValueError):
            e.run_arm(spec,[],cache,model,None,None,times,{},readout='invalid')
