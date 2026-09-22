"""Check the actual candidate's streaming windows and causal head against batch replay."""
import unittest
import json
import numpy as np
import torch
from active.v17.familiar_live_v17 import FamiliarRecognizer, ROOT
from active.v17.train_unified_streaming_ctc_v17 import rolling_windows
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames
from scripts.evaluate_previous_ctc_approved_v17 import collapse

class FamiliarLiveTest(unittest.TestCase):
    def test_real_checkpoint_stream_matches_batch_including_tail(self):
        torch.set_num_threads(2)
        runtime = FamiliarRecognizer(device='cpu')
        manifest = json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text())
        row = next(r for r in manifest['records'] if r['source']=='local_phrases' and r['role']=='validation')
        with np.load(ROOT/row['feature_path'],allow_pickle=False) as data:
            frames=restore_source_frames(data['landmarks'], data['window_source_ranges']).astype(np.float32)
        # Exercise a non-stride-aligned finish and more than the head's receptive field.
        frames=np.concatenate([frames,frames])[:101]
        windows=rolling_windows(frames,stride=4,window_frames=8)
        with torch.inference_mode():
            logits,pooled=runtime.encoder(torch.from_numpy(windows),return_embeddings=True)
            expected=runtime.head(torch.cat((pooled,logits),-1).unsqueeze(0))[0].numpy()
        actual=[]
        for frame in frames:
            if runtime.add(frame) is not None:
                actual.append(runtime.last_logits.copy())
        if runtime.processed_count != runtime.frame_count:
            runtime.finish();actual.append(runtime.last_logits.copy())
        np.testing.assert_allclose(np.stack(actual),expected,atol=2e-4,rtol=2e-4)
        self.assertEqual(runtime.tokens,collapse(expected.argmax(-1)))
        runtime.reset();self.assertIsNone(runtime.finish())
        with self.assertRaises(ValueError): runtime.add(np.full((61,5),np.nan))

if __name__=='__main__': unittest.main()
