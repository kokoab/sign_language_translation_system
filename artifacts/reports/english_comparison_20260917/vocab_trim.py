"""English-focused SentencePiece trimming for the locked mT5 comparison."""
from collections import Counter
import json
from pathlib import Path
import re
import string
from urllib.request import urlretrieve
import zipfile

import torch
from sentencepiece import sentencepiece_model_pb2
from transformers import T5Tokenizer


TARGET_VOCAB_SIZE = 64_000
PREFIX = "Translate sign language video to English: "
BROWN_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
ROOT = Path(__file__).resolve().parents[3]
BROWN_ZIP = ROOT / "data/local/english_text_comparison_20260917/brown.zip"
TOKEN_ID_KEYS = {
    "bos_token_id", "decoder_start_token_id", "eos_token_id", "forced_bos_token_id",
    "forced_eos_token_id", "pad_token_id", "unk_token_id",
}


def _brown_documents(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".zip.part")
        urlretrieve(BROWN_URL, partial)
        partial.replace(path)
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if re.fullmatch(r"brown/c[a-z]\d\d", name))
        if not names:
            raise ValueError(f"no Brown corpus documents in {path}")
        for name in names:
            tagged = archive.read(name).decode("latin-1")
            yield " ".join(token.rsplit("/", 1)[0] for token in tagged.split())


def _remap_token_ids(payload, old_to_new):
    for key in TOKEN_ID_KEYS & payload.keys():
        value = payload[key]
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        try:
            mapped = [old_to_new[int(token)] for token in values]
        except KeyError as error:
            raise ValueError(f"required {key} token {error.args[0]} was not retained") from error
        payload[key] = mapped if isinstance(value, list) else mapped[0]


def prepare_vocab(original_tokenizer_dir, training_texts, output_dir):
    """Build an English-focused tokenizer and explicit old/new row mapping."""
    original_tokenizer_dir, output_dir = Path(original_tokenizer_dir), Path(output_dir)
    training_texts = [str(text) for text in training_texts]
    if not training_texts:
        raise ValueError("training_texts must be nonempty")

    tokenizer = T5Tokenizer.from_pretrained(original_tokenizer_dir, local_files_only=True, legacy=False)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString((original_tokenizer_dir / "spiece.model").read_bytes())
    target = min(TARGET_VOCAB_SIZE, len(proto.pieces))

    required = set(tokenizer.all_special_ids)
    required.update(index for index, piece in enumerate(proto.pieces) if piece.type != piece.NORMAL)
    coverage = string.printable + "£€—“”‘’"
    protected_texts = [*training_texts, PREFIX, coverage, *coverage]
    for text in protected_texts:
        required.update(tokenizer(text, add_special_tokens=False)["input_ids"])
    if len(required) > target:
        raise ValueError(f"{len(required)} required pieces exceed target vocabulary {target}")

    corpus_counts = Counter()
    for document in _brown_documents(BROWN_ZIP):
        corpus_counts.update(tokenizer(document, add_special_tokens=False)["input_ids"])
    selected = set(required)
    for token, _ in corpus_counts.most_common():
        if len(selected) == target:
            break
        selected.add(token)

    english_piece = re.compile(r"[▁A-Za-z0-9'.,!?;:\-()]+\Z")
    ranked = sorted(range(len(proto.pieces)), key=lambda token: proto.pieces[token].score, reverse=True)
    for token in ranked:
        if len(selected) == target:
            break
        if english_piece.fullmatch(proto.pieces[token].piece):
            selected.add(token)
    for token in ranked:
        if len(selected) == target:
            break
        selected.add(token)

    new_to_old = sorted(selected)
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    reduced = sentencepiece_model_pb2.ModelProto()
    reduced.CopyFrom(proto)
    del reduced.pieces[:]
    for old in new_to_old:
        reduced.pieces.add().CopyFrom(proto.pieces[old])
    reduced.trainer_spec.vocab_size = len(new_to_old)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spiece.model").write_bytes(reduced.SerializeToString())
    for source in original_tokenizer_dir.glob("*.json"):
        payload = json.loads(source.read_text())
        if source.name == "config.json":
            payload["vocab_size"] = len(new_to_old)
            _remap_token_ids(payload, old_to_new)
        elif source.name == "generation_config.json":
            _remap_token_ids(payload, old_to_new)
        (output_dir / source.name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    sentinel_ids = {
        piece.piece: {"old": old, "new": old_to_new[old]}
        for old, piece in enumerate(proto.pieces)
        if "<extra_id_" in piece.piece
    }
    with zipfile.ZipFile(BROWN_ZIP) as archive:
        brown_documents = sum(bool(re.fullmatch(r"brown/c[a-z]\d\d", name)) for name in archive.namelist())
    metadata = {
        "source_vocab_size": len(proto.pieces),
        "new_vocab_size": len(new_to_old),
        "old_to_new": {str(old): new for old, new in old_to_new.items()},
        "new_to_old": new_to_old,
        "sentinel_ids": sentinel_ids,
        "brown_url": BROWN_URL,
        "brown_documents": brown_documents,
    }
    (output_dir / "vocab_mapping.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    reduced_tokenizer = T5Tokenizer.from_pretrained(output_dir, local_files_only=True, legacy=False)
    for text in protected_texts:
        old_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        expected = [old_to_new[token] for token in old_ids]
        actual = reduced_tokenizer(text, add_special_tokens=False)["input_ids"]
        if actual != expected or reduced_tokenizer.decode(actual) != tokenizer.decode(old_ids):
            raise ValueError(f"trimmed tokenizer changed protected text tokenization: {text!r}")
    return metadata


def trim_model(model, prepared_dir):
    """Trim a DirectTranslation wrapper's mT5 rows in place and remap its prefix."""
    prepared_dir = Path(prepared_dir)
    mapping = json.loads((prepared_dir / "vocab_mapping.json").read_text())
    new_to_old = torch.tensor(mapping["new_to_old"], dtype=torch.long)
    old_to_new = {int(old): new for old, new in mapping["old_to_new"].items()}
    text_model = getattr(model, "text", model)
    input_embedding = text_model.get_input_embeddings()
    lm_head = text_model.get_output_embeddings()
    if lm_head is None or lm_head.weight.data_ptr() == input_embedding.weight.data_ptr():
        raise ValueError("trim_model requires a separate untied LM head")
    input_rows = input_embedding.weight.detach().index_select(0, new_to_old.to(input_embedding.weight.device)).clone()
    output_rows = lm_head.weight.detach().index_select(0, new_to_old.to(lm_head.weight.device)).clone()
    output_bias = None if lm_head.bias is None else lm_head.bias.detach().index_select(
        0, new_to_old.to(lm_head.bias.device)
    ).clone()

    text_model.resize_token_embeddings(len(new_to_old))
    with torch.no_grad():
        text_model.get_input_embeddings().weight.copy_(input_rows)
        text_model.get_output_embeddings().weight.copy_(output_rows)
        if output_bias is not None:
            text_model.get_output_embeddings().bias.copy_(output_bias)

    config = json.loads((prepared_dir / "config.json").read_text())
    generation = json.loads((prepared_dir / "generation_config.json").read_text())
    for key in TOKEN_ID_KEYS:
        if key in config:
            setattr(text_model.config, key, config[key])
        if key in generation and getattr(text_model, "generation_config", None) is not None:
            setattr(text_model.generation_config, key, generation[key])
    text_model.config.vocab_size = len(new_to_old)

    if hasattr(model, "prefix_ids"):
        try:
            remapped = [old_to_new[int(token)] for token in model.prefix_ids.detach().cpu().flatten()]
        except KeyError as error:
            raise ValueError(f"prefix token {error.args[0]} was not retained") from error
        model.prefix_ids = torch.tensor(remapped, dtype=model.prefix_ids.dtype, device=model.prefix_ids.device).reshape(
            model.prefix_ids.shape
        )
    return model
