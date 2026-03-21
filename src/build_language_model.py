#!/usr/bin/env python3
"""
build_language_model.py - Build bigram language model for CTC beam search rescoring

Reads the manifest to derive canonical vocabulary, builds unigram + bigram counts
from Stage 3 training data and common ASL patterns, then saves as a pickle file
in the exact format camera_inference.py's GlossNGramLM.load() expects.

Usage:
    python src/build_language_model.py

Output:
    weights/gloss_bigram_lm.pkl - Pickle file loadable by GlossNGramLM.load()
"""

import ast
import json
import os
import pickle
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class NgramLanguageModel:
    """Simple N-gram language model for CTC beam search rescoring."""

    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()
        self.total_unigrams = 0

    def train(self, sequences):
        """Train on list of gloss sequences (lists of glosses)."""
        for seq in sequences:
            padded = ['<BOS>'] * (self.n - 1) + list(seq) + ['<EOS>']
            for gloss in seq:
                self.vocab.add(gloss)
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1
            for gloss in seq:
                self.total_unigrams += 1

    def log_prob(self, word, context):
        """Get log probability of word given context (with Laplace smoothing)."""
        context = tuple(context[-(self.n-1):]) if len(context) >= self.n - 1 else tuple(['<BOS>'] * (self.n - 1 - len(context)) + list(context))
        ngram = context + (word,)
        count = self.ngram_counts.get(ngram, 0) + 1
        total = self.context_counts.get(context, 0) + len(self.vocab) + 1
        return np.log(count / total)

    def save(self, path):
        """Save model to JSON file."""
        data = {
            'n': self.n,
            'ngram_counts': {str(k): v for k, v in self.ngram_counts.items()},
            'context_counts': {str(k): v for k, v in self.context_counts.items()},
            'vocab': list(self.vocab),
            'total_unigrams': self.total_unigrams
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load model from JSON file."""
        with open(path) as f:
            data = json.load(f)
        model = cls(n=data['n'])
        model.ngram_counts = {ast.literal_eval(k): v for k, v in data['ngram_counts'].items()}
        model.context_counts = {ast.literal_eval(k): v for k, v in data['context_counts'].items()}
        model.vocab = set(data['vocab'])
        model.total_unigrams = data['total_unigrams']
        return model


def load_manifest_vocab():
    """Load canonical vocabulary from manifest.json."""
    manifest_path = "ASL_landmarks_float16/manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Warning: {manifest_path} not found. Run extraction first.")
        return set(), {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    vocab = set(manifest.values())
    print(f"  Loaded {len(vocab)} canonical labels from manifest")
    return vocab, manifest


def extract_gloss_sequences(manifest_vocab):
    """Extract gloss sequences from Stage 3 CSV and manifest, using canonical labels only."""
    sequences = []

    # Check for Stage 3 CSV data
    stage3_csv = "slt_stage3_dataset_v2.csv"
    if os.path.exists(stage3_csv):
        print(f"Loading gloss sequences from {stage3_csv}...")
        try:
            import csv
            with open(stage3_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "gloss" in row and row["gloss"].strip():
                        glosses = row["gloss"].strip().split()
                        # Only use glosses that are in the manifest vocab
                        if manifest_vocab and all(g in manifest_vocab for g in glosses):
                            sequences.append(glosses)
            print(f"  Found {len(sequences)} valid sequences from Stage 3 CSV")
        except Exception as e:
            print(f"  Error reading Stage 3 CSV: {e}")

    # Check for Stage 3 JSON training data
    stage3_path = "STAGE3_TRAIN/train.json"
    if os.path.exists(stage3_path):
        print(f"Loading gloss sequences from {stage3_path}...")
        with open(stage3_path) as f:
            data = json.load(f)
        count = 0
        for entry in data:
            if "gloss" in entry:
                glosses = entry["gloss"].strip().split()
                if manifest_vocab and all(g in manifest_vocab for g in glosses):
                    sequences.append(glosses)
                    count += 1
        print(f"  Found {count} valid sequences")

    # Add single-word sequences from manifest vocab for unigram coverage
    if manifest_vocab:
        for gloss in sorted(manifest_vocab):
            sequences.append([gloss])
        print(f"  Added {len(manifest_vocab)} single-word sequences from manifest")

    # Common ASL patterns using CANONICAL labels only
    print("Generating common multi-word patterns (canonical labels)...")
    common_patterns = [
        ["HELLO", "HOW", "YOU"],
        ["HOW", "YOU"],
        ["MY", "NAME"],
        ["THANK", "YOU"],
        ["PLEASE", "HELP"],
        ["I", "LOVE", "YOU"],
        ["NICE", "MEET", "YOU"],
        ["GOOD", "MORNING"],
        ["GOOD", "NIGHT"],
        ["WHAT", "YOUR", "NAME"],
        ["WHERE", "YOU", "FROM"],
        ["I", "UNDERSTAND"],
        ["I", "NOT", "UNDERSTAND"],
        ["YES", "PLEASE"],
        ["NO", "THANK", "YOU"],
        # Canonical versions (no composite labels)
        ["I", "DRIVE_CAR", "STORE"],
        ["I", "DRIVE_CAR", "HOME"],
        ["MY", "DRIVE_CAR"],
        ["I", "EAT_FOOD"],
        ["I", "WANT", "EAT_FOOD"],
        ["I", "NEED", "EAT_FOOD"],
        ["EAT_FOOD", "GOOD"],
        ["I", "MAKE_CREATE"],
        ["I", "MAKE_CREATE", "EAT_FOOD"],
        ["HARD_DIFFICULT"],
        ["VERY", "HARD_DIFFICULT"],
        ["NOT", "HARD_DIFFICULT"],
        ["ALSO"],
        ["HE", "GO"],
        ["I", "WANT"],
        ["WE", "GO"],
        ["STORE", "WHERE"],
    ]

    # Filter patterns: only include if ALL glosses are in manifest vocab
    valid_patterns = 0
    for pattern in common_patterns:
        if not manifest_vocab or all(g in manifest_vocab for g in pattern):
            for _ in range(5):
                sequences.append(pattern)
            valid_patterns += 1
    print(f"  Added {valid_patterns} valid common patterns (filtered by manifest vocab)")

    return sequences


def build_bigram_counts(sequences):
    """Build unigram and bigram counts from sequences for GlossNGramLM format."""
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(lambda: defaultdict(int))
    vocab = set()
    total_unigrams = 0

    for seq in sequences:
        prev = '<s>'
        for gloss in seq:
            vocab.add(gloss)
            unigram_counts[gloss] += 1
            bigram_counts[prev][gloss] += 1
            total_unigrams += 1
            prev = gloss
        bigram_counts[prev]['</s>'] += 1

    # Add BOS/EOS to vocab for completeness
    vocab.add('<s>')
    vocab.add('</s>')

    return dict(unigram_counts), {k: dict(v) for k, v in bigram_counts.items()}, vocab, total_unigrams


def save_pickle_lm(unigram_counts, bigram_counts, vocab, total_unigrams, smoothing, path):
    """Save LM in the exact format GlossNGramLM.load() expects."""
    data = {
        'unigram_counts': unigram_counts,
        'bigram_counts': bigram_counts,
        'vocab': vocab,
        'total_unigrams': total_unigrams,
        'smoothing': smoothing,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved pickle LM to {path}")


def main():
    print("=" * 60)
    print("Building Bigram Language Model (GlossNGramLM-compatible)")
    print("=" * 60)

    # Step 1: Load canonical vocab from manifest
    manifest_vocab, manifest = load_manifest_vocab()

    # Step 2: Extract sequences using canonical labels
    sequences = extract_gloss_sequences(manifest_vocab)

    if not sequences:
        print("Error: No gloss sequences found!")
        print("Make sure ASL_landmarks_float16/manifest.json exists (run extraction first)")
        return

    # Step 3: Build bigram counts
    print("\nBuilding bigram language model...")
    unigram_counts, bigram_counts, vocab, total_unigrams = build_bigram_counts(sequences)

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Total unigrams: {total_unigrams}")
    print(f"  Unique bigram contexts: {len(bigram_counts)}")

    # Step 4: Validate vocab alignment with manifest
    if manifest_vocab:
        lm_glosses = {g for g in vocab if g not in ('<s>', '</s>')}
        missing_from_lm = manifest_vocab - lm_glosses
        extra_in_lm = lm_glosses - manifest_vocab
        if missing_from_lm:
            print(f"\n  WARNING: {len(missing_from_lm)} manifest glosses missing from LM: {sorted(missing_from_lm)[:10]}...")
        if extra_in_lm:
            print(f"  WARNING: {len(extra_in_lm)} LM glosses not in manifest: {sorted(extra_in_lm)[:10]}...")
        if not missing_from_lm and not extra_in_lm:
            print("  Vocab alignment check: PASSED (LM vocab == manifest vocab)")

    # Step 5: Save as pickle (primary output — what camera_inference.py loads)
    smoothing = 0.1
    pickle_path = "weights/gloss_bigram_lm.pkl"
    save_pickle_lm(unigram_counts, bigram_counts, vocab, total_unigrams, smoothing, pickle_path)

    # Step 6: Also save JSON trigram for backward compat / debugging
    print("\nAlso training 3-gram model (JSON backup)...")
    lm = NgramLanguageModel(n=3)
    lm.train(sequences)
    os.makedirs("models", exist_ok=True)
    json_path = "models/gloss_lm.json"
    lm.save(json_path)
    print(f"  Saved JSON trigram LM to {json_path}")

    # Step 7: Verify the pickle loads correctly
    print("\nVerification: loading pickle LM...")
    try:
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        assert 'unigram_counts' in loaded
        assert 'bigram_counts' in loaded
        assert 'vocab' in loaded
        assert 'total_unigrams' in loaded
        assert 'smoothing' in loaded
        print(f"  PASSED: Pickle loads correctly, vocab={len(loaded['vocab'])}, unigrams={loaded['total_unigrams']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Step 8: Test a few sequences
    print("\nTest bigram log-probs:")
    test_seqs = [
        ["HELLO"],
        ["HELLO", "HOW", "YOU"],
        ["THANK", "YOU"],
    ]
    for seq in test_seqs:
        if all(g in vocab for g in seq):
            score = 0.0
            prev = '<s>'
            for g in seq:
                bc = bigram_counts.get(prev, {}).get(g, 0)
                pc = unigram_counts.get(prev, 0)
                if pc == 0:
                    prob = (unigram_counts.get(g, 0) + smoothing) / (total_unigrams + smoothing * len(vocab))
                else:
                    prob = (bc + smoothing) / (pc + smoothing * len(vocab))
                score += np.log(prob + 1e-10)
                prev = g
            perplexity = np.exp(-score / len(seq))
            print(f"  {' '.join(seq)}: perplexity = {perplexity:.2f}")
        else:
            oov = [g for g in seq if g not in vocab]
            print(f"  {' '.join(seq)}: [OOV: {oov}]")


if __name__ == "__main__":
    main()
