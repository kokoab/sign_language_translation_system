"""Pinned paired frozen-versus-adapted joint CTC experiment; prepare before train."""
from __future__ import annotations
import argparse, copy, json, os, random, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn.functional as F
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from active.v17.joint_ctc_v17 import Chunks, JointCTC
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from scripts.build_combined_dataset_v17 import load_features
from scripts.train_youtube_motion_pilot_v17 import align_tokens

RECIPE_PATH = ROOT / "active/v17/combined_frozen_joint_manifest_20260922.json"
CODE_PATHS = (Path(__file__), ROOT / "active/v17/approved_phrase_data_v17.py", ROOT / "active/v17/joint_ctc_v17.py", ROOT / "active/v17/model_v17.py", ROOT / "active/v17/model_unified_streaming_ctc_v17.py", ROOT / "active/v17/stage1_window_v17.py", ROOT / "active/v17/train_unified_streaming_ctc_v17.py", ROOT / "scripts/build_combined_dataset_v17.py", ROOT / "scripts/finalize_supplements_v17.py", ROOT / "scripts/train_youtube_motion_pilot_v17.py")


def code_hashes():
    dependencies = (ROOT / "active/v17/model_streaming_stage1_head_v17.py", ROOT / "active/v17/schema_v17.py")
    return {str(path.relative_to(ROOT)): digest(path) for path in CODE_PATHS + dependencies}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def cached_chunks(features):
    """Cached windows are already normalized; dummy record-local indexes carry no clock claim."""
    values = np.asarray(features, np.float32)
    if values.ndim != 4 or values.shape[1:] != (32, 61, 5) or not np.isfinite(values).all():
        raise ValueError("invalid cached 32-frame windows")
    return Chunks(values, [np.arange(32) for _ in values], [np.ones(32, bool) for _ in values], list(range(len(values))))


def paired_plan(single_ids, phrase_ids, seed, epoch):
    if not single_ids or not phrase_ids:
        raise ValueError("both single and phrase training records are required")
    if len(set(single_ids) | set(phrase_ids)) != len(single_ids) + len(phrase_ids):
        raise ValueError("training record IDs must be unique")
    rng = random.Random((seed << 16) + epoch)
    singles, phrases = list(single_ids), list(phrase_ids)
    rng.shuffle(singles)
    cycles = []
    while len(cycles) < len(singles):
        cycle = phrases[:]
        rng.shuffle(cycle)
        cycles.extend(cycle)
    return {"singles": singles, "phrases": cycles[:len(singles)]}


def configure_arm(base, arm):
    if arm not in ("frozen", "adapted"):
        raise ValueError("unknown arm")
    base.requires_grad_(arm == "adapted")
    base.eval()  # identical no-dropout base policy in both arms


def weighted_ctc_loss(logits, targets, lengths, weights):
    if len(targets) != len(lengths) or len(targets) != len(weights) or logits.shape[0] != len(targets):
        raise ValueError("CTC batch mismatch")
    for target, length in zip(targets, lengths):
        required = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        if not target or required > length or length > logits.shape[1] or any(t < 1 or t > 101 for t in target):
            raise ValueError("invalid CTC target")
    flat = torch.tensor([t for target in targets for t in target], dtype=torch.long)
    losses = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1).cpu(), flat,
                        torch.tensor(lengths), torch.tensor([len(t) for t in targets]),
                        blank=0, reduction="none", zero_infinity=False)
    weights = torch.tensor(weights, dtype=losses.dtype)
    losses = losses / torch.tensor([len(target) for target in targets], dtype=losses.dtype)
    return (losses * weights).sum().to(logits.device) / weights.sum().to(logits.device)


def collapse_path(path):
    output, previous = [], None
    for value in path:
        value = int(value)
        if value != previous and 1 <= value <= 101: output.append(value)
        previous = value
    return output


def load_recipe():
    recipe = json.loads(RECIPE_PATH.read_text())
    fixed = {"format": "combined_frozen_joint_v17", "training_entrypoint": "scripts/train_combined_frozen_joint_v17.py",
             "seeds": [17421, 17422], "epochs": 12, "phrase_batch": 4, "single_batch": 32,
             "head_lr": .002, "encoder_lr": .00003, "weight_decay": .0001, "identity_weight": .25}
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise ValueError("fixed recipe contract mismatch")
    for key in ("combined_manifest", "combined_manifest_sha256", "checkpoint", "checkpoint_sha256", "report_dir", "model_dir"):
        if not recipe.get(key): raise ValueError("missing recipe pin: " + key)
    if digest(ROOT / recipe["contract"]) != recipe.get("contract_sha256"):
        raise ValueError("training contract hash mismatch")
    if recipe.get("code_sha256") != code_hashes():
        raise ValueError("recipe code hash mismatch")
    return recipe


def paths(recipe):
    return ROOT / recipe["combined_manifest"], ROOT / recipe["checkpoint"], ROOT / recipe["report_dir"], ROOT / recipe["model_dir"]


def validate_inputs(recipe):
    manifest_path, checkpoint_path, _, _ = paths(recipe)
    if digest(manifest_path) != recipe["combined_manifest_sha256"] or digest(checkpoint_path) != recipe["checkpoint_sha256"]:
        raise ValueError("recipe input hash mismatch")
    verify_manifest()  # required canonical phrase admission check
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "combined_supervised_dataset_v17" or manifest.get("counts") != {"train": 4547, "validation": 1874}:
        raise ValueError("combined manifest contract mismatch")
    labels = manifest.get("label_to_index")
    if not isinstance(labels, dict) or len(labels) != 100 or set(labels.values()) != set(range(100)):
        raise ValueError("frozen vocabulary mismatch")
    for name, expected in manifest.get("source_manifests_sha256", {}).items():
        if digest(ROOT / name) != expected: raise ValueError("source manifest hash mismatch: " + name)
        source_manifest = json.loads((ROOT / name).read_text())
        for evidence, evidence_sha in source_manifest.get("evidence_sha256", {}).items():
            if digest(ROOT / evidence) != evidence_sha: raise ValueError("nested evidence hash mismatch: " + evidence)
    raw = {}
    blocked = {"test", "devtest", "test-clean", "test-other", "protected"}
    for row in manifest["records"]:
        components = set(Path(row["feature_path"]).parts) | set(Path(row["video_path"]).parts)
        if row.get("role") not in ("train", "validation") or components & blocked:
            raise ValueError("protected role/path admitted")
        if not row.get("ctc_targets") or 0 in row["ctc_targets"]: raise ValueError("blank target banned")
        video, sha = row["video_path"], row["video_sha256"]
        if video not in raw: raw[video] = digest(ROOT / video)
        if raw[video] != sha: raise ValueError("raw video hash mismatch: " + video)
        load_features(row, labels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("label_to_index") != labels: raise ValueError("checkpoint labels differ from combined manifest")
    return manifest, checkpoint, labels


def make_records(manifest, labels):
    overlap = {item["feature_path"]: item for item in manifest.get("baseline_parent_overlap", [])}
    for item in overlap.values():
        if item.get("role") not in ("train", "validation") or item.get("role") not in item.get("baseline_roles", []):
            raise ValueError("shared-parent role mismatch")
    result = defaultdict(list)
    for row in manifest["records"]:
        features, targets = load_features(row, labels)
        features = np.asarray(features, dtype=np.float32)
        weight = .5 if row["feature_path"] in overlap else 1.0
        result[row["role"]].append(dict(id=row["feature_path"], source=row["source"], role=row["role"],
            representation=row["representation"], features=features, targets=tuple(targets), weight=weight,
            single=row["supervision"] == "single_verified_sign", label=targets[0] - 1))
    if len({r["id"] for rows in result.values() for r in rows}) != sum(len(rows) for rows in result.values()):
        raise ValueError("record IDs must be unique feature paths")
    return dict(result)


def make_model(checkpoint, device, arm, head_state=None):
    config = checkpoint.get("landmark_model_config", checkpoint.get("model_config"))
    state = checkpoint.get("landmark_model_state_dict", checkpoint.get("model_state_dict"))
    if not config or not state: raise ValueError("checkpoint has no strict landmark model state")
    base = SLTStage1V17(Stage1V17Config(**config))
    base.load_state_dict(state, strict=True)
    configure_arm(base, arm)
    model = JointCTC(base).to(device)
    if head_state is not None: model.head.load_state_dict(head_state, strict=True)
    model.base.eval()
    return model


def forward_records(model, records):
    return model.sequences([cached_chunks(r["features"]) for r in records])


def identity_loss(model, records, device):
    windows = np.concatenate([r["features"] for r in records])
    sizes = [len(r["features"]) for r in records]
    logits = []
    for x in torch.from_numpy(windows).to(device).split(32): logits.append(model.base(x))
    pooled, cursor = [], 0
    for size in sizes:
        pooled.append(torch.cat(logits)[cursor:cursor + size].mean(0)); cursor += size
    values = torch.stack(pooled)
    target = torch.tensor([r["label"] for r in records], device=device)
    weight = torch.tensor([r["weight"] for r in records], device=device)
    return (F.cross_entropy(values, target, reduction="none") * weight).sum() / weight.sum()


def evaluate(model, records, device):
    model.base.eval(); model.head.eval(); out = defaultdict(lambda: dict(samples=0, exact=0, substitutions=0, deletions=0, insertions=0, known_tokens=0, ctc_exact=0, base_top1_correct=0, singles=0))
    with torch.inference_mode():
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            logits, lengths = forward_records(model, batch)
            single_records = [r for r in batch if r["single"]]
            base_predictions = {}
            if single_records:
                windows = torch.from_numpy(np.concatenate([r["features"] for r in single_records])).to(device)
                base_logits = model.base(windows)
                cursor = 0
                for record in single_records:
                    size = len(record["features"])
                    base_predictions[record["id"]] = int(base_logits[cursor:cursor + size].mean(0).argmax().item())
                    cursor += size
            for record, length, value in zip(batch, lengths, logits):
                path = value[:length].argmax(-1).cpu().numpy()
                predicted = collapse_path(path); expected = [x for x in record["targets"] if x != 101]
                bucket = out[record["source"]]; bucket["samples"] += 1; bucket["exact"] += predicted == list(record["targets"]); bucket["known_tokens"] += len(expected)
                edits, _ = align_tokens(expected, [x for x in predicted if x != 101])
                for key, count in edits.items(): bucket[key] += count
                if record["single"]:
                    bucket["singles"] += 1; bucket["ctc_exact"] += int(predicted == [record["label"] + 1])
                    bucket["base_top1_correct"] += int(base_predictions[record["id"]] == record["label"])
    for bucket in out.values():
        bucket["known_wer"] = (bucket["substitutions"] + bucket["deletions"] + bucket["insertions"]) / max(1, bucket["known_tokens"])
        bucket["single_ctc_exact"] = bucket["ctc_exact"] / max(1, bucket["singles"])
        bucket["base_pooled_top1"] = bucket["base_top1_correct"] / max(1, bucket["singles"])
    return dict(out)


def prepare():
    recipe = load_recipe(); manifest_path, checkpoint_path, report, _ = paths(recipe)
    cache = report / "cache.pt"
    if cache.exists() or (report / "preflight.json").exists(): raise FileExistsError("prepare output already exists")
    manifest, checkpoint, labels = validate_inputs(recipe); records = make_records(manifest, labels)
    if {key: len(value) for key, value in records.items()} != {"train": 4547, "validation": 1874}: raise ValueError("record count mismatch")
    if sum(r["single"] for r in records["train"]) != 4264: raise ValueError("single-sign training count mismatch")
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS is required")
    device = torch.device("mps")
    torch.set_num_threads(2)
    report.mkdir(parents=True, exist_ok=True); torch.mps.set_per_process_memory_fraction(.35)
    all_windows = np.concatenate([r["features"] for rows in records.values() for r in rows])
    if len(all_windows) != 7367: raise ValueError("cached window count mismatch")
    gradients = {}
    for arm in ("frozen", "adapted"):
        model = make_model(checkpoint, device, arm); model.head.train(); model.base.eval()
        with torch.no_grad():
            for x in torch.from_numpy(all_windows).split(32):
                if not torch.isfinite(model.tokens(x.to(device))).all(): raise ValueError("nonfinite model forward")
        singles = sorted((r for r in records["train"] if r["single"]), key=lambda r: len(r["features"]), reverse=True)[:32]
        phrases = sorted((r for r in records["train"] if not r["single"]), key=lambda r: len(r["features"]), reverse=True)[:4]
        sl, sn = forward_records(model, singles); pl, pn = forward_records(model, phrases)
        loss = weighted_ctc_loss(sl, [r["targets"] for r in singles], sn, [r["weight"] for r in singles]) + weighted_ctc_loss(pl, [r["targets"] for r in phrases], pn, [r["weight"] for r in phrases]) + identity_loss(model, singles, device) * .25
        loss.backward()
        grads = [p.grad for p in model.base.parameters()]
        if arm == "frozen" and any(g is not None for g in grads): raise ValueError("frozen base has gradients")
        if arm == "adapted" and (any(g is None or not torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads)): raise ValueError("adapted base gradient invalid")
        head_grads = [p.grad for p in model.head.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in head_grads) or not any(g.abs().sum() > 0 for g in head_grads): raise ValueError("head gradient invalid")
        gradients[arm] = dict(loss=float(loss.detach()), base_gradients="absent" if arm == "frozen" else "finite_nonzero", head_gradients="finite_nonzero")
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.head.parameters()): raise ValueError("head lacks finite gradient")
    atomic_torch(cache, dict(records=records, labels=labels))
    preflight = dict(status="passed", recipe_sha256=digest(RECIPE_PATH), code_sha256=code_hashes(), cache_sha256=digest(cache),
        combined_manifest_sha256=digest(manifest_path), checkpoint_sha256=digest(checkpoint_path), counts={k: len(v) for k, v in records.items()},
        source_counts={role: dict(Counter(r["source"] for r in rows)) for role, rows in records.items()}, windows_checked=dict(frozen=7367, adapted=7367), worst_case_batch=dict(singles=32, phrases=4, single_windows=sum(len(r["features"]) for r in singles), phrase_windows=sum(len(r["features"]) for r in phrases)), gradient_checks=gradients, optimizer_steps=0)
    atomic(report / "preflight.json", preflight); return preflight


def train():
    recipe = load_recipe(); _, checkpoint_path, report, models = paths(recipe); preflight = json.loads((report / "preflight.json").read_text())
    cache = report / "cache.pt"
    if not recipe.get("training_ready") or recipe.get("code_sha256") != code_hashes(): raise ValueError("recipe is not training-ready or code is unpinned")
    if preflight.get("status") != "passed" or preflight.get("recipe_sha256") != digest(RECIPE_PATH) or preflight.get("code_sha256") != code_hashes() or preflight.get("cache_sha256") != digest(cache): raise ValueError("preflight pin mismatch")
    if models.exists(): raise FileExistsError("model output already exists")
    manifest, checkpoint, labels = validate_inputs(recipe); payload = torch.load(cache, map_location="cpu", weights_only=False); records = payload["records"]
    if payload["labels"] != labels: raise ValueError("cache labels mismatch")
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS is required")
    models.mkdir(parents=True); device = torch.device("mps"); torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.35)
    results = {}; atomic(report / "status.json", {"state": "running"})
    for seed in recipe["seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        initial = make_model(checkpoint, device, "frozen").head.state_dict()
        for arm in ("frozen", "adapted"):
            model = make_model(checkpoint, device, arm, initial); optimizer = torch.optim.AdamW([
                {"params": model.head.parameters(), "lr": recipe["head_lr"]}, {"params": model.base.parameters(), "lr": recipe["encoder_lr"]}], weight_decay=recipe["weight_decay"])
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            train_rows = records["train"]; singles = [r for r in train_rows if r["single"]]; phrases = [r for r in train_rows if not r["single"]]
            history = []; baseline = evaluate(model, records["validation"], device)
            best = None
            for epoch in range(1, recipe["epochs"] + 1):
                model.head.train(); model.base.eval(); plan = paired_plan([r["id"] for r in singles], [r["id"] for r in phrases], seed, epoch); lookup = {r["id"]: r for r in train_rows}; losses = []; phrase_losses = []; single_losses = []; ce_losses = []
                if len(plan["singles"]) != 4264 or set(plan["singles"]) != {r["id"] for r in singles}: raise ValueError("single-sign epoch coverage mismatch")
                phrase_visits = Counter()
                for start in range(0, len(plan["singles"]), recipe["single_batch"]):
                    sb = [lookup[x] for x in plan["singles"][start:start + recipe["single_batch"]]]
                    pb = [lookup[plan["phrases"][(start // recipe["single_batch"] * recipe["phrase_batch"] + index) % len(plan["phrases"])]] for index in range(recipe["phrase_batch"])]
                    phrase_visits.update(r["id"] for r in pb)
                    sl, sn = forward_records(model, sb); pl, pn = forward_records(model, pb)
                    single_loss = weighted_ctc_loss(sl, [r["targets"] for r in sb], sn, [r["weight"] for r in sb]); phrase_loss = weighted_ctc_loss(pl, [r["targets"] for r in pb], pn, [r["weight"] for r in pb]); ce_loss = identity_loss(model, sb, device); loss = single_loss + phrase_loss + recipe["identity_weight"] * ce_loss
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach())); phrase_losses.append(float(phrase_loss.detach())); single_losses.append(float(single_loss.detach())); ce_losses.append(float(ce_loss.detach()))
                metrics = evaluate(model, records["validation"], device); score = (metrics["local_phrases"]["known_wer"] + metrics["asllrp_contiguous"]["known_wer"]) / 2
                coverage = dict(
                    singles=dict(visits=len(singles), unique=len(set(plan["singles"])), repeats=0,
                                 source_counts=dict(Counter(r["source"] for r in singles))),
                    phrases=dict(visits=sum(phrase_visits.values()), unique=len(phrase_visits),
                                 repeats=sum(phrase_visits.values()) - len(phrase_visits),
                                 source_counts=dict(Counter(lookup[key]["source"] for key, count in phrase_visits.items() for _ in range(count)))),
                )
                row = dict(epoch=epoch, loss=sum(losses) / len(losses), coverage=coverage, validation=metrics,
                           selection_score=score, losses=dict(phrase_ctc=sum(phrase_losses)/len(phrase_losses), single_ctc=sum(single_losses)/len(single_losses), identity_ce=sum(ce_losses)/len(ce_losses)))
                history.append(row); atomic(report / f"history_seed_{seed}_{arm}.json", history)
                if best is None or score < best["score"]: best = dict(score=score, epoch=epoch, state=copy.deepcopy(model.state_dict()), validation=metrics)
            output = models / f"seed_{seed}_{arm}.pth"; atomic_torch(output, dict(format="combined_frozen_joint_v17", arm=arm, seed=seed, base_checkpoint=str(checkpoint_path.relative_to(ROOT)), base_checkpoint_sha256=digest(checkpoint_path), label_to_index=labels, ctc_blank_index=0, other_index=101, base_state_dict={k[5:]: v for k, v in best["state"].items() if k.startswith("base.")}, head_state_dict={k[5:]: v for k, v in best["state"].items() if k.startswith("head.")}, recipe=recipe, selected_epoch=best["epoch"], initial_validation=baseline, selected_validation=best["validation"]))
            saved = torch.load(output, map_location="cpu", weights_only=False)
            selected = make_model(checkpoint, device, arm)
            selected.base.load_state_dict(saved["base_state_dict"], strict=True); selected.head.load_state_dict(saved["head_state_dict"], strict=True)
            selected_train = evaluate(selected, records["train"], device)
            results[f"{seed}:{arm}"] = dict(checkpoint=str(output.relative_to(ROOT)), checkpoint_sha256=digest(output), initial_validation=baseline, history=history, selected_epoch=best["epoch"], selected_validation=best["validation"], selected_train=selected_train); atomic(report / "results.json", {"results": results})
    atomic(report / "status.json", {"state": "complete"}); return results


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--prepare", action="store_true"); group.add_argument("--train", action="store_true"); args = parser.parse_args()
    state = "failed"
    try:
        result = prepare() if args.prepare else train()
        state = "complete"
        return result
    except Exception:
        recipe = json.loads(RECIPE_PATH.read_text()) if RECIPE_PATH.exists() else {}; report = ROOT / recipe.get("report_dir", "artifacts/reports/combined_frozen_joint_v17_20260922")
        atomic(report / "status.json", {"state": "failed", "traceback": traceback.format_exc()}); raise
    finally:
        if args.train:
            subprocess.run(["/usr/bin/osascript", "-e", f'display notification "SLT combined joint {state}; see status.json" with title "SLT"'], capture_output=True)


if __name__ == "__main__": main()
