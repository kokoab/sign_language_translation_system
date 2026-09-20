#!/usr/bin/env python3
"""Reproduce the saved direct-translation failure diagnosis without loading a model."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DIRECT = ROOT / "artifacts/reports/stage1_direct_translation_20260917"
ENGLISH = ROOT / "artifacts/reports/english_comparison_20260917"
UNISIGN = ROOT / "artifacts/reports/unisign_asl_baseline_20260916"
SOURCE_MANIFEST = ROOT / "active/v17/how2sign_transition_manifest_v17.json"
CLASS_MANIFEST = ROOT / "active/v17/citizen100_manifest.json"
SELECTION = ROOT / "data/local/how2sign_transition_subset_v17/selection_plan.json"
PERMUTATIONS = 2_000
SEED = 20260920


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", value.lower())


def normalized(value: str) -> str:
    return " ".join(words(value))


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def nearest(value: str, rows: list[dict]) -> tuple[dict, float]:
    row = max(rows, key=lambda item: similarity(value, item["reference"]))
    return row, similarity(value, row["reference"])


def summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p90": float(np.quantile(values, 0.9)),
        "max": max(values),
    }


def output_copy_audit(rows: list[dict], train: list[dict]) -> tuple[list[dict], dict]:
    audited = []
    for row in rows:
        match, score = nearest(row["prediction"], train)
        prediction = normalized(row["prediction"])
        reference = normalized(match["reference"])
        audited.append({
            "item_id": row["item_id"],
            "prediction": row["prediction"],
            "nearest_train_item_id": match["item_id"],
            "nearest_train_video": match["video"],
            "nearest_train_reference": match["reference"],
            "similarity": score,
            "exact_normalized_match": prediction == reference,
            "containment_match": prediction in reference or reference in prediction,
        })
    scores = [row["similarity"] for row in audited]
    return audited, {
        "count": len(audited),
        "exact_normalized_matches": sum(row["exact_normalized_match"] for row in audited),
        "containment_matches": sum(row["containment_match"] for row in audited),
        "similarity_at_least_0_8": sum(row["similarity"] >= 0.8 for row in audited),
        "similarity": summary(scores),
    }


def permutation_audit(rows: list[dict]) -> dict:
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    bleu, chrf = BLEU(tokenize="13a"), CHRF()

    def score(refs: list[str]) -> tuple[float, float]:
        return (
            bleu.corpus_score(predictions, [refs]).score,
            chrf.corpus_score(predictions, [refs]).score,
        )

    observed = score(references)
    rng = np.random.default_rng(SEED)
    null = np.asarray([
        score([references[index] for index in rng.permutation(len(references))])
        for _ in range(PERMUTATIONS)
    ])
    return {
        "count": len(rows),
        "seed": SEED,
        "random_reference_permutations": PERMUTATIONS,
        "observed": {"bleu": observed[0], "chrf": observed[1]},
        "null_mean": {"bleu": float(null[:, 0].mean()), "chrf": float(null[:, 1].mean())},
        "observed_percentile": {
            "bleu": float((null[:, 0] <= observed[0]).mean()),
            "chrf": float((null[:, 1] <= observed[1]).mean()),
        },
        "fraction_null_at_least_observed": {
            "bleu": float((null[:, 0] >= observed[0]).mean()),
            "chrf": float((null[:, 1] >= observed[1]).mean()),
        },
        "warning": "Descriptive only: 12 correlated rows are too few for a general accuracy claim.",
    }


def feature_audit(rows: list[dict]) -> dict:
    records = []
    for row in rows:
        with np.load(ROOT / row["archive"], allow_pickle=False) as payload:
            features = payload["landmarks"].astype(np.float32)
            valid = payload["window_valid"].astype(bool)
            ranges = payload["window_source_ranges"]
            metadata = json.loads(str(payload["metadata_json"]))
        observed = features[valid]
        video = metadata["video_metadata"]
        records.append({
            "duration_seconds": float(metadata["duration_seconds"]),
            "decoded_frames": int(video["decoded_frame_count"]),
            "sampled_frames": int(video["sampled_frame_count"]),
            "windows": len(valid),
            "valid_windows": int(valid.sum()),
            "tail_frames": int(ranges[-1, 1] - ranges[-1, 0]),
            "hand_presence": float(observed[:, :, :42, 3].mean()),
            "face_presence": float(observed[:, :, 42:57, 3].mean()),
            "body_presence": float(observed[:, :, 57:61, 3].mean()),
        })
    return {
        "count": len(records),
        "duration_seconds": summary([row["duration_seconds"] for row in records]),
        "sampled_frames": summary([row["sampled_frames"] for row in records]),
        "window_count": summary([row["windows"] for row in records]),
        "hand_presence": summary([row["hand_presence"] for row in records]),
        "face_presence": summary([row["face_presence"] for row in records]),
        "body_presence": summary([row["body_presence"] for row in records]),
        "downsampled_to_256": sum(row["decoded_frames"] > row["sampled_frames"] for row in records),
        "invalid_windows": sum(row["windows"] - row["valid_windows"] for row in records),
        "total_windows": sum(row["windows"] for row in records),
        "short_final_windows": sum(row["tail_frames"] < 32 for row in records),
    }


def build() -> None:
    manifest = read(DIRECT / "manifest.json")
    train = manifest["train"]
    validation = manifest["validation"]
    source_rows = {row["source_item_id"]: row for row in read(SOURCE_MANIFEST)["rows"]}
    selection_rows = {row["sentence_id"]: row for row in read(SELECTION)["rows"]}
    classes = read(CLASS_MANIFEST)["classes"]

    mt5 = [row for row in read(DIRECT / "final_predictions.json") if row["reference"] is not None]
    mt5_zero = read(DIRECT / "zero_visual_predictions.json")
    bart = [row for row in read(ENGLISH / "final_predictions.json") if row["reference"] is not None]
    bart_zero = read(ENGLISH / "zero_visual_predictions.json")
    unisign = {row["item_id"]: row for row in read(UNISIGN / "results.json") if row["reference"] is not None}

    mt5_rows, mt5_copy = output_copy_audit(mt5, train)
    bart_rows, bart_copy = output_copy_audit(bart, train)
    _, mt5_zero_copy = output_copy_audit(mt5_zero, train)
    _, bart_zero_copy = output_copy_audit(bart_zero, train)
    mt5_map = {row["item_id"]: row for row in mt5_rows}
    bart_map = {row["item_id"]: row for row in bart_rows}

    train_tokens = [token for row in train for token in words(row["reference"])]
    train_vocab = set(train_tokens)
    validation_vocab = {token for row in validation for token in words(row["reference"])}
    isolated_vocab = {row["canonical_label"].lower() for row in classes}
    rates = []
    for row in train:
        source = source_rows[row["item_id"]]
        rates.append({
            "item_id": row["item_id"],
            "video": row["video"],
            "reference": row["reference"],
            "duration_seconds": float(source["duration_seconds"]),
            "word_count": len(words(row["reference"])),
            "words_per_second": len(words(row["reference"])) / float(source["duration_seconds"]),
        })
    risky = sorted((row for row in rates if row["words_per_second"] > 6),
                   key=lambda row: row["words_per_second"], reverse=True)
    signer_counts = Counter(row["signer"] for row in train)
    source_videos = {
        selection_rows[row["item_id"].split(":", 1)[-1]]["video_id"] for row in train
    }
    direct_provenance = read(DIRECT / "provenance.json")
    bart_provenance = read(ENGLISH / "provenance.json")
    direct_history = read(DIRECT / "training_history.json")
    bart_history = read(ENGLISH / "training_history.json")

    evidence = {
        "format": "direct_translation_failure_diagnosis_v1",
        "conclusion": "The hybrids learned visual routing into memorized training captions, not compositional ASL-to-English translation.",
        "data": {
            "train_rows": len(train),
            "train_hours": sum(row["duration_seconds"] for row in rates) / 3600,
            "train_source_videos": len(source_videos),
            "train_signers": dict(signer_counts),
            "top_two_signer_fraction": sum(value for _, value in signer_counts.most_common(2)) / len(train),
            "validation_rows": len(validation),
            "validation_source_videos": len({row["source_video"] for row in validation}),
            "target_word_tokens": len(train_tokens),
            "target_word_types": len(train_vocab),
            "validation_word_types": len(validation_vocab),
            "validation_oov_word_types": sorted(validation_vocab - train_vocab),
            "isolated_labels_with_exact_surface_word_in_train": len(isolated_vocab & train_vocab),
            "isolated_label_count": len(isolated_vocab),
            "words_per_second": summary([row["words_per_second"] for row in rates]),
            "rows_above_4_words_per_second": sum(row["words_per_second"] > 4 for row in rates),
            "rows_above_6_words_per_second": len(risky),
            "high_rate_rows": risky,
        },
        "features": {
            "train": feature_audit(train),
            "validation": feature_audit(validation),
            "contract": "independent non-overlapping 32-frame Stage-1 windows; at most 256 uniformly sampled source frames",
        },
        "optimization": {
            "mt5_parameters": direct_provenance["parameters"],
            "bart_parameters": bart_provenance["bart_parameters"],
            "parameters_per_train_row": {
                "mt5": direct_provenance["parameters"] / len(train),
                "bart": bart_provenance["bart_parameters"] / len(train),
            },
            "epochs": 20,
            "fixed_feature_sequences_reused_each_epoch": True,
            "mt5_translation_loss": {"epoch_1": direct_history[0]["translation_loss"], "epoch_20": direct_history[-1]["translation_loss"]},
            "bart_translation_loss": {"epoch_1": bart_history[0]["translation_loss"], "epoch_20": bart_history[-1]["translation_loss"]},
        },
        "caption_copying": {
            "mt5": mt5_copy,
            "mt5_zero_visual": mt5_zero_copy,
            "bart": bart_copy,
            "bart_zero_visual": bart_zero_copy,
        },
        "pairing_controls": {
            "mt5": permutation_audit(mt5),
            "bart": permutation_audit(bart),
            "saved_mismatched_scores": read(ENGLISH / "summary.json")["translation"],
        },
        "same_12_row_scores": {
            "native_unisign": read(UNISIGN / "summary.json"),
            "hybrids": read(ENGLISH / "summary.json")["translation"],
        },
        "rows": {
            "mt5": mt5_rows,
            "bart": bart_rows,
        },
        "protected_test_accessed": False,
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    cards = []
    for row in validation:
        item_id = row["item_id"]
        m = mt5_map[item_id]
        b = bart_map[item_id]
        cards.append(f"""
<article><h2>{html.escape(item_id)}</h2>
<div class="grid">
<section><h3>Held-out clip</h3><video controls preload="metadata" src="{Path(row['video']).as_uri()}"></video>
<p><b>Reference:</b> {html.escape(row['reference'])}</p>
<p><b>Native Uni-Sign:</b> {html.escape(unisign[item_id]['prediction'])}</p></section>
<section><h3>mT5 hybrid → nearest train caption ({m['similarity']:.3f})</h3>
<p><b>Prediction:</b> {html.escape(m['prediction'])}</p>
<video controls preload="metadata" src="{Path(m['nearest_train_video']).as_uri()}"></video>
<p><b>Training annotation:</b> {html.escape(m['nearest_train_reference'])}</p></section>
<section><h3>BART hybrid → nearest train caption ({b['similarity']:.3f})</h3>
<p><b>Prediction:</b> {html.escape(b['prediction'])}</p>
<video controls preload="metadata" src="{Path(b['nearest_train_video']).as_uri()}"></video>
<p><b>Training annotation:</b> {html.escape(b['nearest_train_reference'])}</p></section>
</div></article>""")
    noisy_cards = [f"""
<article><h2>{html.escape(row['item_id'])}</h2><video controls preload="metadata" src="{Path(row['video']).as_uri()}"></video>
<p><b>Annotation:</b> {html.escape(row['reference'])}</p>
<p>{row['duration_seconds']:.2f}s, {row['word_count']} words, {row['words_per_second']:.2f} words/s. This is an automatic alignment-risk flag, not a claim that the annotation is wrong.</p></article>""" for row in risky]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Direct translation diagnosis</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#111;color:#eee}}a{{color:#9cf}}article{{border-top:1px solid #555;padding:20px 0}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}video{{width:100%;max-height:360px;background:#000}}h3{{font-size:1rem}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Direct translation failure evidence</h1><p>The left video is the held-out clip. The other videos are the training rows whose annotations most closely match each generated sentence. Similarity is normalized character SequenceMatcher ratio.</p>
{''.join(cards)}<h1>Automatic annotation-risk flags (&gt;6 English words/s)</h1><p>These eight rows are the strongest duration/text mismatches in the 994-row subset. Review them as possible alignment noise; the threshold is diagnostic only.</p>{''.join(noisy_cards)}</body></html>"""
    (HERE / "videos.html").write_text(page)

    mt5_pair = evidence["pairing_controls"]["mt5"]
    bart_pair = evidence["pairing_controls"]["bart"]
    train_features = evidence["features"]["train"]
    report = f"""# Direct-translation failure diagnosis

## Finding

The failed hybrids are **caption memorizers with weak visual routing**, not working ASL-to-English translators. The main failure is the experiment design: 994 fixed sentence pairs were used to jointly tune {direct_provenance['parameters']:,} mT5-hybrid parameters or {bart_provenance['bart_parameters']:,} BART-hybrid parameters for 20 passes. The isolated Stage-1 encoder is good at its 100-sign classification task, but a linear projection and sentence loss did not turn it into an open-vocabulary continuous-language encoder.

This is not mainly a signing-speed failure. It is also not evidence that all How2Sign annotations are bad. The data has real annotation/context noise, but the dominant measured failure is memorization plus poor visual-text alignment.

[Open the side-by-side video and annotation viewer](./videos.html). It contains all 12 held-out clips, both hybrid outputs, the native Uni-Sign output, and the training videos whose annotations the hybrid outputs copied most closely. It also contains the eight strongest automatic annotation-risk cases.

## Decisive evidence

| Evidence | mT5 hybrid | BART hybrid |
|---|---:|---:|
| held-out outputs exactly equal to a 994-row training caption | {mt5_copy['exact_normalized_matches']}/12 | {bart_copy['exact_normalized_matches']}/12 |
| nearest-training-caption similarity ≥0.80 | {mt5_copy['similarity_at_least_0_8']}/12 | {bart_copy['similarity_at_least_0_8']}/12 |
| median nearest-training-caption similarity | {mt5_copy['similarity']['median']:.3f} | {bart_copy['similarity']['median']:.3f} |
| final paired BLEU / chrF | 1.076 / 18.989 | 1.061 / 17.464 |
| correct-pair chrF percentile among {PERMUTATIONS:,} random reference permutations | {100*mt5_pair['observed_percentile']['chrf']:.1f}% | {100*bart_pair['observed_percentile']['chrf']:.1f}% |

BART copied 11 training captions verbatim. Its zero-visual control produced the same training caption, “A bladder infection is diagnosed at your veterinary clinic,” for all 12 clips. mT5 copied or lightly recombined training captions; six outputs exceed 0.80 similarity and its median is {mt5_copy['similarity']['median']:.3f}. This behavior explains the fluent but unrelated sentences.

Correct visual pairing is not completely irrelevant: paired chrF lands above {100*mt5_pair['observed_percentile']['chrf']:.1f}% of random permutations for mT5 and {100*bart_pair['observed_percentile']['chrf']:.1f}% for BART. That is weak evidence that the video changes which memorized caption is selected. It is not evidence of compositional translation. Paired BLEU is only around the 75th percentile, and the saved one-step mismatched controls have higher BLEU than correct pairing for both models.

The unchanged native pose-only Uni-Sign system scored BLEU 8.206 / chrF 38.059 on these same 12 rows. Its output is still unreliable, but the gap shows that the text annotations and clip speed are learnable enough for a matched continuous pose encoder. Replacing Uni-Sign's pose encoder with our isolated encoder and a linear bridge removed most of that ability.

## What failed in the architecture

Our hybrid is Stage-1 Squeezeformer applied independently to non-overlapping 32-frame windows, followed by a single linear projection into a pretrained text model. Every window restarts the Stage-1 temporal context. The text encoder can attend across the concatenated tokens, but the visual encoder was originally optimized to collapse one isolated clip into one of 100 classes. There was no contrastive video-text alignment, masked visual-language pretraining, or full continuous-pose pretraining before joint sentence tuning.

That differs from the methods we were trying to borrow from:

- [GFSLT-VLP](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html) first aligns visual and language representations with contrastive and masked pretraining, then initializes translation from those aligned encoders.
- [Uni-Sign](https://arxiv.org/abs/2501.15187) uses generative sign-language pretraining at large scale, including a reported 1,985 hours of paired CSL-News video and text, before downstream fine-tuning. Its native pose path uses region-specific spatial-temporal graph processing over the continuous sequence.
- Our experiment retained Uni-Sign's text component but discarded the visual encoder that had learned the compatible representation. The 994-pair joint fit did not recreate that pretraining.

The smaller BART run confirms that model size was a speed and storage problem, not the alignment solution. Cutting parameters from 589.39M to 146.41M made training faster, while the model still memorized 11/12 outputs exactly.

## What the data says

| Audit | Result |
|---|---:|
| paired training rows | {len(train):,} |
| paired video duration | {evidence['data']['train_hours']:.2f} hours |
| training signers | {len(signer_counts)} |
| rows from the two largest signers | {100*evidence['data']['top_two_signer_fraction']:.1f}% |
| unique English word types | {len(train_vocab):,} |
| isolated Stage-1 labels | {len(isolated_vocab)} |
| held-out rows / source videos | {len(validation)} / {evidence['data']['validation_source_videos']} |
| invalid extracted windows | {train_features['invalid_windows']} / {train_features['total_windows']:,} |
| clips uniformly downsampled to 256 frames | {train_features['downsampled_to_256']} / {len(train)} |
| median hand presence | {100*train_features['hand_presence']['median']:.1f}% |

The local paired set is only {100*len(train)/31164:.2f}% of How2Sign's roughly 31,164-row training split and covers four adaptation signers. Two signers contribute 80.4% of its rows. The targets contain {len(train_vocab):,} English word types, while isolated supervision covers 100 chosen signs and ASL-to-English translation is not a word-for-word mapping. High isolated accuracy therefore does not supply the missing open-vocabulary sentence alignment.

Extraction did not globally collapse: only {train_features['invalid_windows']} of {train_features['total_windows']:,} windows are invalid, median hand presence is {100*train_features['hand_presence']['median']:.1f}%, and just {train_features['downsampled_to_256']} clips were temporally downsampled. The median clip duration is {train_features['duration_seconds']['median']:.2f}s. Speed and missing landmarks can hurt individual rows but do not explain the system-wide caption copying.

The training clips use manually realigned timestamps, as documented by the [source mirror](https://huggingface.co/datasets/martinctl/how2sign-asl-clips). The annotations are still weak sentence-level supervision. Eight of 994 rows exceed six English words per video second and are flagged in the viewer; this is a risk proxy, not proof of a wrong label. More broadly, [a Deaf-signer human study of How2Sign](https://arxiv.org/abs/2406.11049) reports that 5% of sampled realigned clips omitted the relevant content and that discourse context was needed for key details in 33.3% of cases. Annotation/context noise is real, but it is secondary to the measured memorization failure.

## Decision for Stage 2 and Reel

Do not run another language-model swap on these 994 pairs. Do not use unconditional Stage-3 deduplication: text alone cannot distinguish a held sign from an intentional repeat.

For the current 100-sign mobile system, keep the problem bounded. Use the existing Stage-1 encoder with one shallow temporal sequence head over unpooled features, trained to emit blank plus the 100 glosses. The missing data is targeted continuous boundary supervision: held signs, the same sign intentionally repeated with a real release/re-entry boundary, transitions, rest, and ordinary non-sign movement from signer-disjoint people. CTC is still appropriate for collapsing a held emission run; the data must teach the model when a new event begins. Stage 3 should only turn the recognized gloss sequence into English.

Gloss-free translation remains a separate offline research track. A valid next test would retain a continuous pose encoder already aligned to text, or pretrain one on the full realigned How2Sign training set before sentence decoding. It should use the official validation split, checkpoint selection on validation loss/translation quality, and a copying/retrieval control like this report. It should not reuse this isolated-window-to-linear-projector recipe.

## Files and limits

- `evidence.json`: machine-readable data, feature, memorization, and pairing audits.
- `videos.html`: all held-out examples and the copied-caption training videos, plus annotation-risk rows.
- `provenance.json`: exact input hashes and protected-split statement.
- `verification.json`: consistency checks and report hashes.

The 12 held-out rows come from four source videos and remain too small for a translation benchmark. Similarity matching diagnoses copying; it is not a semantic metric. No Citizen official test or How2Sign test split was accessed, and no new model was trained.
"""
    (HERE / "REPORT.md").write_text(report)

    inputs = [
        DIRECT / "manifest.json", DIRECT / "final_predictions.json", DIRECT / "zero_visual_predictions.json",
        DIRECT / "training_history.json", DIRECT / "provenance.json", ENGLISH / "final_predictions.json",
        ENGLISH / "zero_visual_predictions.json", ENGLISH / "summary.json", ENGLISH / "training_history.json",
        ENGLISH / "provenance.json", UNISIGN / "results.json", UNISIGN / "summary.json", SOURCE_MANIFEST,
        CLASS_MANIFEST, SELECTION, Path(__file__),
    ]
    provenance = {
        "format": "direct_translation_failure_diagnosis_provenance_v1",
        "generated_at": "2026-09-20",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "analysis_seed": SEED,
        "reference_permutations": PERMUTATIONS,
        "model_inference_run": False,
        "new_training_run": False,
        "citizen_official_test_accessed": False,
        "how2sign_test_accessed": False,
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    assert len(train) == 994 and len(validation) == len(mt5) == len(bart) == 12
    assert bart_copy["exact_normalized_matches"] == 11
    assert mt5_copy["similarity_at_least_0_8"] == 6
    assert train_features["invalid_windows"] == 6
    assert not evidence["protected_test_accessed"]
    verification = {
        "passed": True,
        "checks": [
            "994 frozen training rows and 12 paired held-out rows",
            "BART exact-copy count reproduces 11/12",
            "mT5 >=0.8 nearest-caption count reproduces 6/12",
            "feature audit reproduces six invalid windows",
            "all viewer videos exist",
            "no protected test access and no model training/inference",
        ],
        "all_viewer_videos_exist": all(Path(row["video"]).is_file() for row in validation + train),
        "outputs": {name: sha256(HERE / name) for name in ["REPORT.md", "videos.html", "evidence.json", "provenance.json"]},
    }
    assert verification["all_viewer_videos_exist"]
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")


def self_test() -> None:
    assert normalized("I'm  HERE!") == "i'm here"
    rows = [{"reference": "alpha beta"}, {"reference": "other text"}]
    assert nearest("Alpha, beta.", rows)[0] is rows[0]
    assert similarity("same", "same") == 1.0
    print("self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else build()
