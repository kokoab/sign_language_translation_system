import torch

# Load your Stage 2 checkpoint
checkpoint = torch.load('weights/stage2_best_model.pth', map_location='cpu', weights_only=False)

# Extract the vocabulary
idx_to_gloss = checkpoint['idx_to_gloss']

print(f"Vocabulary loaded! Total signs: {len(idx_to_gloss)}")
# Example: print(idx_to_gloss['1']) -> 'HELLO'