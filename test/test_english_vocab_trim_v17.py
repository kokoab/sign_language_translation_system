import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import MT5Config, MT5ForConditionalGeneration, T5Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/reports/english_comparison_20260917/vocab_trim.py"
TOKENIZER = ROOT / "data/local/unisign_asl_baseline_20260916/mt5-base"


def load_module():
    spec = importlib.util.spec_from_file_location("english_vocab_trim", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnglishVocabTrimTest(unittest.TestCase):
    def test_prepare_and_trim_preserve_required_tokens_and_logits(self):
        trim = load_module()
        prefix = "Translate sign language video to English: "
        training = ["A quiet zephyr crossed Manila.", "The signer said hello twice."]

        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            corpus = temporary / "brown.zip"
            with zipfile.ZipFile(corpus, "w") as archive:
                archive.writestr(
                    "brown/ca01",
                    "The/at quick/jj brown/jj fox/nn jumps/vbz ./.\n"
                    "Clear/jj English/nn sentences/nns remain/vb useful/jj ./.",
                )
            output = temporary / "tokenizer"
            with patch.object(trim, "BROWN_ZIP", corpus), patch.object(
                trim, "TARGET_VOCAB_SIZE", 512
            ):
                prepared = trim.prepare_vocab(TOKENIZER, training, output)

            original = T5Tokenizer.from_pretrained(TOKENIZER, legacy=False)
            reduced = T5Tokenizer.from_pretrained(output, legacy=False)
            mapping = json.loads((output / "vocab_mapping.json").read_text())
            old_to_new = {int(k): v for k, v in mapping["old_to_new"].items()}

            self.assertEqual(len(reduced), 512)
            self.assertEqual(prepared["new_vocab_size"], 512)
            self.assertEqual(reduced.pad_token_id, old_to_new[original.pad_token_id])
            self.assertEqual(reduced.eos_token_id, old_to_new[original.eos_token_id])
            self.assertEqual(reduced.unk_token_id, old_to_new[original.unk_token_id])
            for text in [*training, prefix]:
                old_ids = original(text, add_special_tokens=False)["input_ids"]
                self.assertEqual(
                    reduced(text, add_special_tokens=False)["input_ids"],
                    [old_to_new[token] for token in old_ids],
                )
                self.assertEqual(reduced.decode([old_to_new[token] for token in old_ids]), original.decode(old_ids))
            sentinel = original.convert_tokens_to_ids("▁<extra_id_0>")
            self.assertIn(sentinel, old_to_new)
            self.assertEqual(reduced.convert_ids_to_tokens(old_to_new[sentinel]), "▁<extra_id_0>")

            torch.manual_seed(7)
            config = MT5Config(
                vocab_size=250112,
                d_model=8,
                d_ff=16,
                d_kv=4,
                num_heads=2,
                num_layers=1,
                num_decoder_layers=1,
                dropout_rate=0.0,
                tie_word_embeddings=False,
                decoder_start_token_id=0,
                pad_token_id=0,
                eos_token_id=1,
            )
            class Wrapper(nn.Module):
                def __init__(self, text, prefix_ids):
                    super().__init__()
                    self.text = text
                    self.projection = nn.Linear(3, 3)
                    self.register_buffer("prefix_ids", prefix_ids)

            text_model = MT5ForConditionalGeneration(config)
            prefix_ids = original(prefix, return_tensors="pt")["input_ids"]
            model = Wrapper(text_model, prefix_ids).eval()
            projection_before = model.projection.weight.detach().clone()
            old_input = original(training[0], return_tensors="pt")["input_ids"]
            old_decoder = original(training[1], return_tensors="pt")["input_ids"]
            with torch.no_grad():
                old_logits = model.text(input_ids=old_input, decoder_input_ids=old_decoder).logits

            trim.trim_model(model, output)
            new_input = torch.tensor([[old_to_new[int(token)] for token in old_input[0]]])
            new_decoder = torch.tensor([[old_to_new[int(token)] for token in old_decoder[0]]])
            with torch.no_grad():
                new_logits = model.text(input_ids=new_input, decoder_input_ids=new_decoder).logits
            retained = torch.tensor(mapping["new_to_old"])
            torch.testing.assert_close(new_logits, old_logits.index_select(-1, retained), rtol=0, atol=1e-6)
            self.assertEqual(model.text.config.vocab_size, 512)
            self.assertEqual(model.text.get_input_embeddings().num_embeddings, 512)
            self.assertEqual(model.text.lm_head.out_features, 512)
            self.assertEqual(model.prefix_ids.tolist(), [[old_to_new[int(token)] for token in prefix_ids[0]]])
            torch.testing.assert_close(model.projection.weight, projection_before)


if __name__ == "__main__":
    unittest.main()
