#!/usr/bin/env python3
"""Paired familiar-local-signers CTC head experiment on the earlier causal pipeline."""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames
from active.v17.train_unified_streaming_ctc_v17 import rolling_windows
from scripts.build_combined_dataset_v17 import load_features
from scripts.evaluate_previous_ctc_approved_v17 import collapse
from scripts.train_combined_frozen_joint_v17 import validate_inputs as validate_combined_inputs
from scripts.train_combined_frozen_joint_v17 import code_hashes as combined_code_hashes
from scripts.train_youtube_motion_pilot_v17 import align_tokens

RECIPE_PATH = ROOT / "active/v17/local_familiar_ctc_manifest_20260922.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp); temp.replace(path)


def code_hashes() -> dict[str, str]:
    paths = (Path(__file__), ROOT / "scripts/evaluate_previous_ctc_approved_v17.py",
             ROOT / "scripts/build_combined_dataset_v17.py", ROOT / "active/v17/model_v17.py",
             ROOT / "active/v17/model_unified_streaming_ctc_v17.py",
             ROOT / "active/v17/train_unified_streaming_ctc_v17.py",
             ROOT / "active/v17/train_streaming_tcn_ctc_v17.py")
    return combined_code_hashes() | {str(path.relative_to(ROOT)): digest(path) for path in paths}


def load_recipe() -> dict:
    recipe = json.loads(RECIPE_PATH.read_text())
    fixed = dict(format="local_familiar_ctc_v17", training_entrypoint="scripts/train_local_familiar_ctc_v17.py",
                 seeds=[17521, 17522], epochs=6, phrase_batch=4, single_batch=32,
                 head_lr=.0001, weight_decay=.001)
    if any(recipe.get(k) != v for k, v in fixed.items()):
        raise ValueError("fixed local familiar recipe mismatch")
    required = ("combined_manifest", "combined_manifest_sha256", "split_manifest", "split_manifest_sha256",
                "checkpoint", "checkpoint_sha256", "base_checkpoint", "base_checkpoint_sha256",
                "report_dir", "model_dir", "code_sha256")
    if any(not recipe.get(k) for k in required) or recipe["code_sha256"] != code_hashes():
        raise ValueError("missing or stale recipe pin")
    if digest(ROOT / recipe["contract"]) != recipe.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    return recipe


def load_rows(recipe: dict) -> tuple[dict[str, list[dict]], dict, dict]:
    combined = ROOT / recipe["combined_manifest"]; split = ROOT / recipe["split_manifest"]
    if digest(combined) != recipe["combined_manifest_sha256"] or digest(split) != recipe["split_manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    # Reuse the canonical verifier: every listed feature/video/source-manifest hash is checked.
    manifest, _, labels = validate_combined_inputs(dict(recipe, checkpoint=recipe["base_checkpoint"], checkpoint_sha256=recipe["base_checkpoint_sha256"]))
    assignment = json.loads(split.read_text())
    if len(manifest["records"]) != 6421 or assignment.get("counts") != {"train": 4686, "validation": 1735}:
        raise ValueError("unexpected input record counts")
    rows = {r["feature_path"]: r for r in manifest["records"]}
    assigned = assignment.get("records", [])
    if len(assigned) != len(rows) or {r["feature_path"] for r in assigned} != set(rows):
        raise ValueError("split does not cover the canonical combined rows exactly")
    for item in assigned:
        row = rows[item["feature_path"]]
        if any(item.get(key) != row.get(key) for key in ("source", "feature_sha256", "video_sha256")) or item.get("original_role") != row.get("role"):
            raise ValueError("split row does not match canonical source row")
    roles = {r["feature_path"]: r["experiment_role"] for r in assigned}
    changed = [r for r in assigned if r["experiment_role"] != r["original_role"]]
    if len(changed) != 139 or any(r["source"] != "local_phrases" or r["original_role"] != "validation" or r["experiment_role"] != "train" for r in changed):
        raise ValueError("unexpected familiar signer override")
    local_train = {r["video_sha256"] for r in assigned if r["source"] == "local_phrases" and r["experiment_role"] == "train"}
    local_validation = {r["video_sha256"] for r in assigned if r["source"] == "local_phrases" and r["experiment_role"] == "validation"}
    if local_train & local_validation:
        raise ValueError("local raw-video hash crosses experiment roles")
    output = []
    for path, row in rows.items():
        features, targets = load_features(row, labels)
        features = np.asarray(features, np.float32)
        if row["supervision"] == "approved_phrase_or_subspan":
            with np.load(ROOT / path, allow_pickle=False) as data:
                frames = restore_source_frames(data["landmarks"], data["window_source_ranges"])
            features = rolling_windows(frames, stride=4, window_frames=8)
        if not np.isfinite(features).all() or features.shape[1:] != (32, 61, 5):
            raise ValueError("invalid old-pipeline windows")
        output.append(dict(id=path, source=row["source"], original_role=row["role"], familiar_role=roles[path],
                           features=features, targets=tuple(int(x) for x in targets), single=row["supervision"] == "single_verified_sign",
                           weight=.5 if path in {x["feature_path"] for x in manifest.get("baseline_parent_overlap", [])} else 1.0))
    return {"all": output}, manifest, labels


def paired_plan(singles: list[str], phrases: list[str], seed: int, epoch: int) -> dict[str, list[str]]:
    rng = random.Random((seed << 16) + epoch); singles, phrases = singles[:], phrases[:]
    rng.shuffle(singles); repeated = []
    while len(repeated) < len(singles):
        cycle = phrases[:]; rng.shuffle(cycle); repeated.extend(cycle)
    return {"singles": singles, "phrases": repeated[:len(singles)]}


def ctc_loss(logits: torch.Tensor, records: list[dict], lengths: list[int]) -> torch.Tensor:
    targets = [r["targets"] for r in records]
    if any(not x or len(x) + sum(a == b for a, b in zip(x, x[1:])) > n for x, n in zip(targets, lengths)):
        raise ValueError("infeasible CTC target")
    flat = torch.tensor([x for target in targets for x in target], dtype=torch.long)
    rows = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1).cpu(), flat,
                      torch.tensor(lengths), torch.tensor([len(x) for x in targets]), blank=0,
                      reduction="none", zero_infinity=False)
    rows = rows / torch.tensor([len(x) for x in targets], dtype=rows.dtype)
    weights = torch.tensor([r["weight"] for r in records], dtype=rows.dtype)
    rows = (rows * weights).sum() / weights.sum()
    if not torch.isfinite(rows): raise ValueError("nonfinite CTC loss")
    return rows.to(logits.device)


def forward(head, records: list[dict], evidence: dict[str, torch.Tensor], device: torch.device):
    values = [evidence[r["id"]].clone().to(device, dtype=torch.float32) for r in records]
    return head(pad_sequence(values, batch_first=True)), [len(v) for v in values]


def known(sequence):
    return [int(value) for value in sequence if int(value) != 101]


def metrics(head, records: list[dict], evidence: dict[str, torch.Tensor], device: torch.device) -> dict:
    out = defaultdict(lambda: dict(samples=0, exact=0, substitutions=0, deletions=0, insertions=0, target_tokens=0, predicted_tokens=0))
    was_training = head.training; head.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]; logits, lengths = forward(head, batch, evidence, device)
            for r, n, value in zip(batch, lengths, logits):
                predicted = collapse(value[:n].argmax(-1).cpu().numpy()); expected_known, predicted_known = known(r["targets"]), known(predicted); edits, _ = align_tokens(expected_known, predicted_known)
                b = out[r["source"]]; b["samples"] += 1; b["exact"] += int(predicted == list(r["targets"])); b["target_tokens"] += len(expected_known); b["predicted_tokens"] += len(predicted_known)
                for k, v in edits.items(): b[k] += v
    for b in out.values():
        b["wer"] = (b["substitutions"] + b["deletions"] + b["insertions"]) / max(1, b["target_tokens"]); b["exact_accuracy"] = b["exact"] / max(1, b["samples"])
    if was_training: head.train()
    return dict(out)


def cache_evidence(base, records: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}; base.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]; x = np.concatenate([r["features"] for r in batch])
            pieces = []
            for value in torch.from_numpy(x).split(64):
                logits, pooled = base(value.to(device), return_embeddings=True); pieces.append(torch.cat((pooled, logits), -1).cpu().float())
            values = torch.cat(pieces); cursor = 0
            for r in batch:
                result[r["id"]] = values[cursor:cursor + len(r["features"])]; cursor += len(r["features"])
    if len(result) != 6421 or any(not torch.isfinite(x).all() for x in result.values()): raise ValueError("invalid cached evidence")
    return result


def model(recipe, device):
    head_data = torch.load(ROOT / recipe["checkpoint"], map_location="cpu", weights_only=False)
    base_data = torch.load(ROOT / recipe["base_checkpoint"], map_location="cpu", weights_only=False)
    if digest(ROOT / recipe["checkpoint"]) != recipe["checkpoint_sha256"] or digest(ROOT / recipe["base_checkpoint"]) != recipe["base_checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    if head_data.get("label_to_index") != base_data.get("label_to_index") or head_data.get("ctc_blank_index") != 0 or head_data.get("other_index") != 101:
        raise ValueError("old CTC/base vocabulary or indexes mismatch")
    base = SLTStage1V17(Stage1V17Config(**base_data["model_config"])); base.load_state_dict(base_data["model_state_dict"], strict=True)
    head = load_unified_streaming_head(head_data, device=device); base.to(device).eval().requires_grad_(False)
    return base, head


def subsets(records: list[dict], arm: str):
    role = "original_role" if arm == "control" else "familiar_role"
    train = [r for r in records if r[role] == "train"]
    # The evaluation is held fixed for both arms: 60 retained local signer-02 + 12 ASLLRP + 1663 singles.
    evaluation = [r for r in records if r["single"] and r["original_role"] == "validation"]
    evaluation += [r for r in records if not r["single"] and r["source"] == "asllrp_contiguous" and r["original_role"] == "validation"]
    evaluation += [r for r in records if not r["single"] and r["source"] == "local_phrases" and r["familiar_role"] == "validation"]
    expected = {"single": 1663, "local_phrases": 60, "asllrp_contiguous": 12}
    got = {"single": sum(r["single"] for r in evaluation), "local_phrases": sum(r["source"] == "local_phrases" for r in evaluation), "asllrp_contiguous": sum(r["source"] == "asllrp_contiguous" for r in evaluation)}
    if got != expected: raise ValueError(f"fixed evaluation mismatch: {got}")
    return train, evaluation


def prepare(recipe):
    report = ROOT / recipe["report_dir"]
    if (report / "preflight.json").exists() or (report / "evidence.pt").exists(): raise FileExistsError("refusing preflight overwrite")
    packed, manifest, labels = load_rows(recipe); records = packed["all"]
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS required")
    device = torch.device("mps"); torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.35)
    base, head = model(recipe, device); evidence = cache_evidence(base, records, device)
    report.mkdir(parents=True, exist_ok=True); cache = report / "evidence.pt"; atomic_torch(cache, evidence)
    direct = records[:4]; logits, lengths = forward(head, direct, evidence, device)
    with torch.inference_mode():
        x = torch.from_numpy(np.concatenate([r["features"] for r in direct])).to(device); raw = []
        for v in x.split(64):
            l, p = base(v, return_embeddings=True); raw.append(torch.cat((p, l), -1))
        cursor = 0; direct_values=[]
        for r in direct: direct_values.append(torch.cat(raw)[cursor:cursor+len(r["features"])]); cursor += len(r["features"])
        direct_logits = head(pad_sequence(direct_values, batch_first=True))
    if not torch.allclose(logits, direct_logits, atol=1e-3, rtol=1e-3): raise ValueError("cached evidence differs from direct evidence")
    for start in range(0, len(records), 32):
        all_logits, _ = forward(head, records[start:start + 32], evidence, device)
        if not torch.isfinite(all_logits).all(): raise ValueError("nonfinite cached-head output")
    train, _ = subsets(records, "control"); singles = [r for r in train if r["single"]]; phrases = [r for r in train if not r["single"]]
    sample = singles[:32] + phrases[:4]; head.train()
    sl, sn = forward(head, sample[:32], evidence, device); pl, pn = forward(head, sample[32:], evidence, device); loss = ctc_loss(sl, sample[:32], sn) + ctc_loss(pl, sample[32:], pn); loss.backward()
    grads = [p.grad for p in head.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads) or any(p.grad is not None for p in base.parameters()): raise ValueError("invalid preflight gradients")
    atomic(report / "preflight.json", dict(status="passed", recipe_sha256=digest(RECIPE_PATH), code_sha256=code_hashes(), evidence_sha256=digest(cache), counts=dict(records=len(records), windows=sum(len(r["features"]) for r in records)), fixed_evaluation=dict(local_phrases=60, asllrp_contiguous=12, single=1663), cache_direct_max_delta=float((logits-direct_logits).abs().max()), gradients="head finite/nonzero; base absent", optimizer_steps=0))


def train(recipe):
    report, models = ROOT / recipe["report_dir"], ROOT / recipe["model_dir"]
    preflight = json.loads((report / "preflight.json").read_text())
    if not recipe.get("training_ready") or preflight.get("status") != "passed" or preflight.get("recipe_sha256") != digest(RECIPE_PATH) or preflight.get("code_sha256") != code_hashes() or models.exists(): raise ValueError("training gate/preflight mismatch")
    packed, _, _ = load_rows(recipe); records=packed["all"]; evidence=torch.load(report / "evidence.pt", map_location="cpu", weights_only=False)
    if digest(report / "evidence.pt") != preflight["evidence_sha256"] or not torch.backends.mps.is_available(): raise ValueError("evidence/MPS mismatch")
    device=torch.device("mps"); torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.35); models.mkdir(parents=True); results={}; atomic(report / "status.json", {"state":"running"})
    try:
        for seed in recipe["seeds"]:
            _, initial = model(recipe, device); initial_state=copy.deepcopy(initial.state_dict())
            seed_baseline = None
            for arm in ("control", "familiar"):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                _, head=model(recipe,device); head.load_state_dict(initial_state); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); train_rows, evaluation=subsets(records,arm); singles=[r for r in train_rows if r["single"]]; phrases=[r for r in train_rows if not r["single"]]; lookup={r["id"]:r for r in train_rows}
                expected_train = 4547 if arm == "control" else 4686
                if len(train_rows) != expected_train: raise ValueError("unexpected arm training count")
                optimizer=torch.optim.AdamW(head.parameters(),lr=recipe["head_lr"],weight_decay=recipe["weight_decay"]); baseline=metrics(head,evaluation,evidence,device); best=dict(epoch=0,score=baseline["local_phrases"]["wer"],state=copy.deepcopy(head.state_dict()),validation=baseline); history=[]
                if seed_baseline is None:
                    seed_baseline = baseline
                elif baseline != seed_baseline:
                    raise ValueError("paired epoch-zero validation mismatch")
                for epoch in range(1,recipe["epochs"]+1):
                    head.train(); plan=paired_plan([r["id"] for r in singles],[r["id"] for r in phrases],seed,epoch); visits=Counter(); losses=[]
                    for start in range(0,len(plan["singles"]),32):
                        sb=[lookup[x] for x in plan["singles"][start:start+32]]; offset=(start//32)*4; pb=[lookup[x] for x in plan["phrases"][offset:offset+4]]; visits.update(r["id"] for r in pb)
                        sl,sn=forward(head,sb,evidence,device); pl,pn=forward(head,pb,evidence,device); loss=ctc_loss(sl,sb,sn)+ctc_loss(pl,pb,pn); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach()))
                    validation=metrics(head,evaluation,evidence,device); row=dict(epoch=epoch,loss=sum(losses)/len(losses),selection_local60_wer=validation["local_phrases"]["wer"],validation=validation,coverage=dict(single_visits=len(plan["singles"]),single_unique=len(set(plan["singles"])),phrase_visits=sum(visits.values()),phrase_unique=len(visits),phrase_repeats=sum(visits.values())-len(visits),phrase_source_visits=dict(Counter(lookup[k]["source"] for k,n in visits.items() for _ in range(n)))))
                    history.append(row); atomic(report/f"history_seed_{seed}_{arm}.json",history)
                    if row["selection_local60_wer"] < best["score"]: best=dict(epoch=epoch,score=row["selection_local60_wer"],state=copy.deepcopy(head.state_dict()),validation=validation)
                head.load_state_dict(best["state"]); full_train=metrics(head,train_rows,evidence,device); output=models/f"seed_{seed}_{arm}.pth"; atomic_torch(output,dict(format="local_familiar_ctc_v17",seed=seed,arm=arm,selected_epoch=best["epoch"],head_state_dict=best["state"],selected_validation=best["validation"],full_train=full_train,recipe=recipe)); results[f"{seed}:{arm}"]=dict(checkpoint=str(output.relative_to(ROOT)),checkpoint_sha256=digest(output),initial_validation=baseline,history=history,selected_epoch=best["epoch"],selected_validation=best["validation"],full_train=full_train); atomic(report/"results.json",dict(results=results))
        atomic(report/"status.json",{"state":"complete"})
    except Exception:
        atomic(report/"status.json",{"state":"failed","traceback":traceback.format_exc()}); raise


def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True); group.add_argument("--prepare",action="store_true");group.add_argument("--train",action="store_true");args=parser.parse_args(); recipe={}; state="failed"
    try:
        recipe=load_recipe()
        prepare(recipe) if args.prepare else train(recipe)
        state="complete"
    except Exception:
        report = ROOT / recipe.get("report_dir", "artifacts/reports/local_familiar_signer_v17_20260922")
        atomic(report / "status.json", {"state":"failed", "traceback":traceback.format_exc()})
        raise
    finally:
        if args.train: subprocess.run(["/usr/bin/osascript","-e",f'display notification "SLT local familiar CTC {state}; see status.json" with title "SLT"'],capture_output=True)


if __name__ == "__main__": main()
