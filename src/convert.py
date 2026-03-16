import torch
import torch.nn as nn
from train_stage_2 import SLTStage2CTC


class ExportWrapper(nn.Module):
    """Traceable wrapper for ONNX export. Single input x [B, 32, 42, 10] per clip; no x_lens or batch loop."""

    def __init__(self, model: SLTStage2CTC):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [B, 32, 42, 10] — one or more 32-frame clips (no variable length)
        enc_out = self.model.encoder(x)           # [B, 32, 256]
        enc_out = enc_out.permute(0, 2, 1)       # [B, 256, 32]
        pooled = self.model.temporal_pool(enc_out)  # [B, 256, 4]
        pooled = pooled.permute(0, 2, 1)        # [B, 4, 256]
        lstm_out, _ = self.model.lstm(pooled)    # [B, 4, 1024]
        logits = self.model.classifier(lstm_out) # [B, 4, vocab_size]
        return logits


# Load checkpoint first to get vocab_size (saved by train_stage_2)
checkpoint = torch.load('weights/stage2_best_model.pth', map_location='cpu', weights_only=False)
vocab_size = checkpoint.get('vocab_size')
if vocab_size is None:
    vocab_size = checkpoint['model_state_dict']['classifier.weight'].shape[0]

model = SLTStage2CTC(vocab_size=vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export using traceable wrapper (single input: clips [B, 32, 42, 10])
wrapped = ExportWrapper(model)
dummy_x = torch.randn(1, 32, 42, 10)
torch.onnx.export(
    wrapped,
    dummy_x,
    "weights/stage2_best_model.onnx",
    input_names=["clips"],
    output_names=["logits"],
    dynamic_axes={"clips": {0: "batch"}, "logits": {0: "batch"}},
)
