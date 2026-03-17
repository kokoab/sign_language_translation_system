"""Debug hand-count scoring for FPS=15 N=2 case."""
import os, sys, json, torch, numpy as np, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import test_video_pipeline as tvp
from train_stage_2 import SLTStage2CTC

# Build hand-count lookup
tvp._build_hand_count_lookup()
print('GLOSS_HAND_COUNT sample:')
for g in ['HOW', 'YOU', 'DRINK', 'TOMORROW', 'LOW', 'STOP', 'DO', 'HELLO']:
    print(f'  {g}: {tvp.GLOSS_HAND_COUNT.get(g, "NOT FOUND")} hand(s)')

# Load model
ckpt = torch.load(tvp.STAGE2_CKPT, map_location=tvp.DEVICE, weights_only=False)
idx_to_gloss = ckpt['idx_to_gloss']
vocab_size = ckpt['vocab_size']

model = SLTStage2CTC(vocab_size=vocab_size).to(tvp.DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Extract at FPS=15 normal
xyz_seq, l_ever, r_ever = tvp.extract_landmarks_from_video(tvp.VIDEO_PATH, override_fps=15)
print(f'\nl_ever={l_ever}, r_ever={r_ever}')
print(f'xyz_seq shape: {xyz_seq.shape}')

# Test N=1,2,3,4
max_signs = min(4, max(1, xyz_seq.shape[0] // 10))
for n in range(1, max_signs + 1):
    features, seg_hand_counts = tvp.build_hypothesis(xyz_seq, n, l_ever, r_ever)
    print(f'\n{"="*60}')
    print(f'N={n} | seg_hand_counts: {seg_hand_counts} | features: {features.shape}')
    print(f'{"="*60}')

    beams = tvp._score_hypothesis(model, features, idx_to_gloss)

    print(f'Top 10 beams:')
    for i, (glosses, log_prob) in enumerate(beams[:10]):
        prob = float(np.exp(log_prob))
        gloss_str = ' '.join(glosses)

        # Check length match
        len_match = len(glosses) == len(seg_hand_counts)

        # Apply hand-count prior manually with detail
        if tvp.GLOSS_HAND_COUNT and len_match:
            match_bonus = 1.0
            details = []
            for gloss, observed in zip(glosses, seg_hand_counts):
                expected = tvp.GLOSS_HAND_COUNT.get(gloss, 0)
                if expected == observed:
                    match_bonus *= 1.5
                    details.append(f'{gloss}(exp={expected}==obs={observed} -> 1.5x)')
                elif expected > 0 and expected != observed:
                    match_bonus *= 0.7
                    details.append(f'{gloss}(exp={expected}!=obs={observed} -> 0.7x)')
                else:
                    details.append(f'{gloss}(exp={expected},obs={observed} -> 1.0x)')

            # Also apply length penalty if len(glosses) > n
            len_penalty = 1.0
            if len(glosses) > n:
                len_penalty = n / len(glosses)

            adjusted = prob * match_bonus * len_penalty
            print(f'  #{i+1}: [{gloss_str}] raw={prob:.6f} bonus={match_bonus:.2f} lenpen={len_penalty:.2f} adj={adjusted:.6f}')
            print(f'        {" | ".join(details)}')
        else:
            len_penalty = 1.0
            if len(glosses) > n:
                len_penalty = n / len(glosses)
            adjusted = prob * len_penalty
            reason = "no GLOSS_HAND_COUNT" if not tvp.GLOSS_HAND_COUNT else f"len mismatch ({len(glosses)} vs {len(seg_hand_counts)})"
            print(f'  #{i+1}: [{gloss_str}] raw={prob:.6f} lenpen={len_penalty:.2f} adj={adjusted:.6f} ({reason})')
