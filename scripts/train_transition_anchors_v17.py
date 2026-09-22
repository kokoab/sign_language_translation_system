"""Pinned head-only CTC refinement with approved contextual core/gap anchors."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from active.v17.approved_phrase_data_v17 import digest
from scripts.train_combined_frozen_joint_v17 import (
    cached_chunks, code_hashes as combined_code_hashes, evaluate, make_model, make_records, paired_plan, validate_inputs,
    weighted_ctc_loss,
)

RECIPE_PATH = ROOT / "active/v17/transition_anchors_manifest_20260922.json"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def code_hashes():
    return combined_code_hashes() | {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}


def anchor_loss(logits, record_ids, anchors):
    """Mean each region, then average known and blank groups when both exist."""
    values = defaultdict(list)
    for row, record_id in enumerate(record_ids):
        for anchor in anchors.get(record_id, []):
            indices = anchor["indices"]
            if not indices:
                continue
            target = int(anchor["target"])
            if not 0 <= target <= 100:
                raise ValueError("anchor target must be blank=0 or known=1..100")
            value = F.cross_entropy(logits[row, indices], torch.full((len(indices),), target, device=logits.device, dtype=torch.long))
            values["blank" if target == 0 else "known"].append(value)
    groups = [torch.stack(group).mean() for group in values.values() if group]
    return (torch.stack(groups).mean() if groups else logits.sum() * 0), {key: len(value) for key, value in values.items()}


def load_recipe():
    recipe = json.loads(RECIPE_PATH.read_text())
    fixed = {"format": "transition_anchors_v17", "training_entrypoint": "scripts/train_transition_anchors_v17.py", "seeds": [17421, 17422], "epochs": 3, "phrase_batch": 4, "single_batch": 32, "head_lr": .0001, "weight_decay": .0001, "anchor_weight": .25}
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise ValueError("fixed anchor recipe mismatch")
    for key in ("combined_manifest", "combined_manifest_sha256", "anchors", "anchors_sha256", "report_dir", "model_dir", "checkpoints", "base_checkpoint", "base_checkpoint_sha256", "contract", "contract_sha256"):
        if not recipe.get(key):
            raise ValueError("missing recipe pin: " + key)
    if recipe.get("code_sha256") != code_hashes():
        raise ValueError("recipe code hash mismatch")
    if digest(ROOT / recipe["contract"]) != recipe["contract_sha256"]:
        raise ValueError("contract hash mismatch")
    return recipe


def load_anchors(recipe, records, labels):
    path = ROOT / recipe["anchors"]
    if digest(path) != recipe["anchors_sha256"]:
        raise ValueError("anchor file hash mismatch")
    data = json.loads(path.read_text())
    for name, expected in data.get("inputs", {}).items():
        if digest(ROOT / name) != expected: raise ValueError("anchor input hash mismatch: " + name)
    anchored = data.get("records")
    if not isinstance(anchored, dict):
        raise ValueError("anchors must contain records")
    valid = {record["id"]: record for rows in records.values() for record in rows}
    result = defaultdict(list)
    for record_id, source in anchored.items():
        if record_id not in valid or valid[record_id]["source"] != "asllrp_contiguous" or source.get("role") != valid[record_id]["role"]:
            raise ValueError("anchor does not reference an approved ASLLRP record")
        if source.get("feature_sha256") != digest(ROOT / record_id):
            raise ValueError("anchor feature hash mismatch")
        seen = set()
        for region in source.get("regions", []):
            indices, target = region.get("indices"), region.get("target")
            if not isinstance(indices, list) or not indices or any(not isinstance(index, int) or index < 0 or index >= len(valid[record_id]["features"]) * 32 or index in seen for index in indices):
                raise ValueError("anchor indices must be nonempty, bounded, and disjoint")
            seen.update(indices)
            if target == 0:
                if region.get("kind") != "internal_gap":
                    raise ValueError("only internal-gap anchors may target blank")
            elif target not in range(1, 101) or labels.get(region.get("label")) != target - 1 or region.get("kind") != "sign_core":
                raise ValueError("known anchor label/target/kind mismatch")
            result[record_id].append({"indices": indices, "target": target})
    if not result:
        raise ValueError("no anchor regions")
    return dict(result), data


def cached_evidence(model, records, device):
    output = {}
    model.base.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]
            windows = np.concatenate([record["features"] for record in batch])
            values = []
            for item in torch.from_numpy(windows).split(64):
                value = model.evidence(item.to(device)).cpu()
                if not torch.isfinite(value).all():
                    raise ValueError("nonfinite rich evidence")
                values.append(value)
            values = torch.cat(values)
            cursor = 0
            for record in batch:
                count = len(record["features"])
                output[record["id"]] = values[cursor:cursor + count].reshape(-1, values.shape[-1])
                cursor += count
    return output


def forward_cache(head, records, evidence, device):
    sequences = [evidence[record["id"]].clone().to(device) for record in records]
    lengths = [len(value) for value in sequences]
    return head(pad_sequence(sequences, batch_first=True)), lengths


def preflight(recipe):
    report = ROOT / recipe["report_dir"]
    if (report / "preflight.json").exists() or (report / "cache").exists():
        raise FileExistsError("anchor preflight output exists")
    manifest, base, labels = validate_inputs(dict(recipe, checkpoint=recipe["base_checkpoint"], checkpoint_sha256=recipe["base_checkpoint_sha256"]))
    records = make_records(manifest, labels)
    if sum(len(r['features']) for rows in records.values() for r in rows) != 7367:
        raise ValueError("expected exactly 7367 cached windows")
    for rows in records.values():
        for record in rows:
            if record["features"].shape[1:] != (32, 61, 5) or not np.isfinite(record["features"]).all():
                raise ValueError("invalid cached feature window")
    anchors, anchor_data = load_anchors(recipe, records, labels)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.35)
    device = torch.device("mps")
    gradients, cache_hashes = {}, {}
    for seed in recipe["seeds"]:
        checkpoint_path = ROOT / recipe["checkpoints"][str(seed)]["path"]
        if digest(checkpoint_path) != recipe["checkpoints"][str(seed)]["sha256"]:
            raise ValueError("adapted checkpoint hash mismatch")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = make_model(base, device, "frozen")
        model.base.load_state_dict(saved["base_state_dict"], strict=True); model.head.load_state_dict(saved["head_state_dict"], strict=True)
        model.base.requires_grad_(False); model.base.eval()
        cache = cached_evidence(model, records["train"] + records["validation"], device)
        cache_path = report / "cache" / f"seed_{seed}.pt"; atomic_torch(cache_path, cache); cache_hashes[str(seed)] = digest(cache_path)
        phrase_rows = [record for record in records["train"] if not record["single"]]
        known = next((record for record in phrase_rows if any(item["target"] > 0 for item in anchors.get(record["id"], []))), None)
        blank = next((record for record in phrase_rows if any(item["target"] == 0 for item in anchors.get(record["id"], []))), None)
        if known is None or blank is None: raise ValueError("preflight needs known and blank anchors")
        phrase = [known, blank] + [record for record in phrase_rows if record['id'] not in {known['id'], blank['id']}][:2]
        model.head.eval()
        cached_logits, cached_lengths = forward_cache(model.head, phrase, cache, device)
        direct_logits, direct_lengths = model.sequences([cached_chunks(record["features"]) for record in phrase])
        if cached_lengths != direct_lengths or not torch.allclose(cached_logits, direct_logits, atol=1e-4, rtol=1e-4):
            raise ValueError("cached evidence does not reproduce direct CTC logits")
        cache_delta = float((cached_logits - direct_logits).abs().max().cpu())
        model.head.train()
        logits, lengths = forward_cache(model.head, phrase, cache, device)
        ctc = weighted_ctc_loss(logits, [record["targets"] for record in phrase], lengths, [record["weight"] for record in phrase])
        aux, counts = anchor_loss(logits, [record["id"] for record in phrase], anchors)
        if not counts.get("known") or not counts.get("blank"): raise ValueError("preflight anchor group missing")
        loss = ctc + recipe["anchor_weight"] * aux
        loss.backward()
        head = [parameter.grad for parameter in model.head.parameters()]
        if any(value is None or not torch.isfinite(value).all() for value in head) or not any(value.abs().sum() > 0 for value in head):
            raise ValueError("anchor head gradient invalid")
        if any(parameter.grad is not None for parameter in model.base.parameters()):
            raise ValueError("frozen base received gradient")
        gradients[str(seed)] = {"loss": float(loss.detach()), "anchor_regions": counts, "cache_max_logit_delta": cache_delta, "windows_checked": 7367, "head_gradients": "finite_nonzero", "base_gradients": "absent", "optimizer_steps": 0}
    atomic(report / "preflight.json", {"status": "passed", "recipe_sha256": digest(RECIPE_PATH), "code_sha256": code_hashes(), "anchors_sha256": digest(ROOT / recipe["anchors"]), "cache_sha256": cache_hashes, "counts": {role: len(rows) for role, rows in records.items()}, "anchor_regions": sum(len(item["regions"]) for item in anchor_data["records"].values()), "gradient_checks": gradients, "optimizer_steps": 0})


def train(recipe):
    report, models = ROOT / recipe["report_dir"], ROOT / recipe["model_dir"]
    preflight_data = json.loads((report / "preflight.json").read_text())
    if not recipe.get("training_ready") or preflight_data.get("status") != "passed" or preflight_data.get("recipe_sha256") != digest(RECIPE_PATH) or preflight_data.get("code_sha256") != code_hashes() or models.exists():
        raise ValueError("training gate/preflight/model output mismatch")
    manifest, base, labels = validate_inputs(dict(recipe, checkpoint=recipe["base_checkpoint"], checkpoint_sha256=recipe["base_checkpoint_sha256"]))
    records = make_records(manifest, labels); anchors, _ = load_anchors(recipe, records, labels)
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS is required")
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.35); device = torch.device("mps"); models.mkdir(parents=True)
    all_results = {}; atomic(report / "status.json", {"state": "running"})
    try:
        for seed in recipe["seeds"]:
            checkpoint_path = ROOT / recipe["checkpoints"][str(seed)]["path"]
            if digest(checkpoint_path) != recipe["checkpoints"][str(seed)]["sha256"]: raise ValueError("adapted checkpoint hash mismatch")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cache_path = report / "cache" / f"seed_{seed}.pt"
            if digest(cache_path) != preflight_data["cache_sha256"][str(seed)]: raise ValueError("evidence cache hash mismatch")
            evidence = torch.load(cache_path, map_location="cpu", weights_only=False)
            initial = make_model(base, device, "frozen"); initial.base.load_state_dict(saved["base_state_dict"], strict=True); initial.head.load_state_dict(saved["head_state_dict"], strict=True); initial.base.requires_grad_(False); initial.base.eval()
            for arm in ("control", "anchored"):
                model = copy.deepcopy(initial); model.head.train(); optimizer = torch.optim.AdamW(model.head.parameters(), lr=recipe["head_lr"], weight_decay=recipe["weight_decay"])
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                base_state = {key: value.detach().cpu().clone() for key, value in model.base.state_dict().items()}
                baseline = evaluate(model, records["validation"], device); best = dict(epoch=0, score=(baseline["local_phrases"]["known_wer"] + baseline["asllrp_contiguous"]["known_wer"]) / 2, state=copy.deepcopy(model.head.state_dict()), validation=baseline); history=[]
                singles, phrases = [r for r in records["train"] if r["single"]], [r for r in records["train"] if not r["single"]]; lookup={r["id"]:r for r in records["train"]}
                for epoch in range(1, 4):
                    model.head.train(); model.base.eval(); plan=paired_plan([r["id"] for r in singles],[r["id"] for r in phrases],seed,epoch); losses=[]; ctc_losses=[]; aux_losses=[]; exposures=Counter(); phrase_visits=Counter()
                    for start in range(0,len(plan["singles"]),32):
                        sb=[lookup[x] for x in plan["singles"][start:start+32]]; pb=[lookup[plan["phrases"][(start//32*4+i)%len(plan["phrases"])] ] for i in range(4)]
                        phrase_visits.update(record["id"] for record in pb)
                        sl,sn=forward_cache(model.head,sb,evidence,device); pl,pn=forward_cache(model.head,pb,evidence,device)
                        loss=weighted_ctc_loss(sl,[r["targets"] for r in sb],sn,[r["weight"] for r in sb])+weighted_ctc_loss(pl,[r["targets"] for r in pb],pn,[r["weight"] for r in pb])
                        aux, counts=anchor_loss(pl,[r["id"] for r in pb],anchors)
                        if arm=="anchored": loss=loss+recipe["anchor_weight"]*aux
                        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.head.parameters(),5.,error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach())); ctc_losses.append(float((loss-recipe["anchor_weight"]*aux if arm=="anchored" else loss).detach())); aux_losses.append(float(aux.detach())); exposures.update(counts)
                    validation=evaluate(model,records["validation"],device); score=(validation["local_phrases"]["known_wer"]+validation["asllrp_contiguous"]["known_wer"])/2; history.append(dict(epoch=epoch,loss=sum(losses)/len(losses),losses=dict(ctc=sum(ctc_losses)/len(ctc_losses),anchor=sum(aux_losses)/len(aux_losses)),anchor_regions=dict(exposures),coverage=dict(singles=len(singles),phrase_visits=sum(phrase_visits.values()),phrase_unique=len(phrase_visits)),validation=validation,selection_score=score)); atomic(report/f"history_seed_{seed}_{arm}.json",history)
                    if score<best["score"]: best=dict(epoch=epoch,score=score,state=copy.deepcopy(model.head.state_dict()),validation=validation)
                if any(not torch.equal(value, model.base.state_dict()[key].detach().cpu()) for key, value in base_state.items()): raise ValueError("frozen base weights changed")
                model.head.load_state_dict(best["state"]); selected_train=evaluate(model,records["train"],device); output=models/f"seed_{seed}_{arm}.pth"; atomic_torch(output,dict(format="transition_anchors_v17",seed=seed,arm=arm,selected_epoch=best["epoch"],base_checkpoint=recipe["checkpoints"][str(seed)],head_state_dict=best["state"],selected_validation=best["validation"],selected_train=selected_train,recipe=recipe)); all_results[f"{seed}:{arm}"]=dict(checkpoint=str(output.relative_to(ROOT)),checkpoint_sha256=digest(output),history=history,selected_epoch=best["epoch"],selected_validation=best["validation"],selected_train=selected_train); atomic(report/"results.json",{"results":all_results})
        atomic(report/"status.json",{"state":"complete"})
    except Exception:
        atomic(report/"status.json",{"state":"failed","traceback":traceback.format_exc()}); raise


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--prepare", action="store_true"); group.add_argument("--train", action="store_true"); args = parser.parse_args()
    recipe = load_recipe()
    if args.prepare:
        try: preflight(recipe)
        except Exception:
            atomic(ROOT / recipe["report_dir"] / "status.json", {"state": "preflight_failed", "traceback": traceback.format_exc()})
            raise
    else: train(recipe)


if __name__ == "__main__":
    state = "failed"
    try:
        main()
        state = "complete"
    except Exception:
        atomic(ROOT / "artifacts/reports/transition_anchors_v17_20260922/status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        if "--train" in sys.argv:
            subprocess.run(["/usr/bin/osascript", "-e", f'display notification "SLT transition anchors {state}; see status.json" with title "SLT"'], capture_output=True)
