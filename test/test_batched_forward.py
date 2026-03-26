"""Verify that the batched SLTStage2CTC.forward produces identical output
to the original per-sample loop version."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.nn.utils.rnn import pad_sequence

def original_forward(model, x, x_lens):
    """Original per-sample loop forward (copy of old code)."""
    B = x.size(0)
    out_seqs, out_lens = [], []
    V, C = x.shape[2], x.shape[3]
    for b in range(B):
        valid_x = x[b, :x_lens[b]]
        num_clips = valid_x.size(0) // 32
        remainder = valid_x.size(0) % 32
        if remainder > 0:
            pad_frames = torch.zeros(32 - remainder, V, C, device=valid_x.device)
            valid_x = torch.cat([valid_x, pad_frames], dim=0)
            num_clips += 1
        clips = valid_x.view(num_clips, 32, V, C)
        with torch.no_grad():
            enc_out = model.encoder(clips)
        enc_out = enc_out.permute(0, 2, 1)
        pooled = model.temporal_pool(enc_out)
        pooled = pooled.permute(0, 2, 1)
        seq_features = pooled.reshape(num_clips * 4, -1)
        out_seqs.append(seq_features)
        out_lens.append(num_clips * 4)
    padded_seqs = pad_sequence(out_seqs, batch_first=True)
    max_len = padded_seqs.size(1)
    padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= torch.tensor(out_lens, device=x.device).unsqueeze(1)
    seq_out = model.seq_transformer(padded_seqs, padding_mask=padding_mask)
    logits = model.classifier(seq_out)
    out_lens_t = torch.tensor(out_lens, dtype=torch.long, device=x.device)
    return logits, out_lens_t


def test_batched_forward():
    from src.train_stage_2 import SLTStage2CTC

    device = 'cpu'
    model = SLTStage2CTC(vocab_size=311, stage1_ckpt=None, d_model=128, seq_layers=2)
    model.to(device)
    model.eval()

    # Create variable-length batch: 4 samples with different lengths
    # Sample 0: 64 frames (2 clips), Sample 1: 96 frames (3 clips)
    # Sample 2: 48 frames (2 clips with padding), Sample 3: 128 frames (4 clips)
    lengths = [64, 96, 48, 128]
    max_len = max(lengths)
    B = len(lengths)
    x = torch.randn(B, max_len, 47, 16)
    x_lens = torch.tensor(lengths, dtype=torch.long)

    with torch.no_grad():
        # New batched forward
        logits_new, lens_new = model(x, x_lens)

        # Original per-sample forward
        logits_old, lens_old = original_forward(model, x, x_lens)

    # Verify shapes match
    assert logits_new.shape == logits_old.shape, f"Shape mismatch: {logits_new.shape} vs {logits_old.shape}"
    assert torch.equal(lens_new, lens_old), f"Lens mismatch: {lens_new} vs {lens_old}"

    # Verify values match (allow tiny floating point differences)
    max_diff = (logits_new - logits_old).abs().max().item()
    assert max_diff < 1e-4, f"Output mismatch: max diff = {max_diff}"

    print(f"PASSED: Shapes match: {logits_new.shape}")
    print(f"PASSED: Lens match: {lens_new.tolist()}")
    print(f"PASSED: Max diff = {max_diff:.2e} (< 1e-4)")
    print(f"PASSED: Batched forward is mathematically identical to original.")

    # Also test with return_inter=True
    logits_new_i, lens_new_i, inter_new = model(x, x_lens, return_inter=True)
    assert logits_new_i.shape == logits_new.shape, "InterCTC forward shape mismatch"
    assert inter_new is not None, "InterCTC logits should not be None"
    print(f"PASSED: InterCTC forward works. Inter shape: {inter_new.shape}")

    # Test with odd-length sequences (remainder > 0)
    lengths_odd = [33, 65, 50, 97]
    max_len_odd = max(lengths_odd)
    x_odd = torch.randn(4, max_len_odd, 47, 16)
    x_lens_odd = torch.tensor(lengths_odd, dtype=torch.long)
    with torch.no_grad():
        logits_odd_new, lens_odd_new = model(x_odd, x_lens_odd)
        logits_odd_old, lens_odd_old = original_forward(model, x_odd, x_lens_odd)
    max_diff_odd = (logits_odd_new - logits_odd_old).abs().max().item()
    assert max_diff_odd < 1e-4, f"Odd-length mismatch: max diff = {max_diff_odd}"
    print(f"PASSED: Odd-length sequences. Max diff = {max_diff_odd:.2e}")

    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    test_batched_forward()
