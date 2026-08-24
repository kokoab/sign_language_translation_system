#!/usr/bin/env python3
"""Cache pre-pooled MobileCLIP2 hand-view spatial maps for efficient fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path: sys.path.insert(0, str(repo_root))
    from active.v17.extract_hand_mobileclip2_v17 import load_crop_archive
    from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
    from active.v17.schema_hand_spatial_mobileclip2_v17 import HandSpatialMobileCLIP2V17Config, schema_fingerprint, schema_payload
else:
    from .extract_hand_mobileclip2_v17 import load_crop_archive
    from .extract_mobileclip2_v17 import build_encoder, select_device
    from .schema_hand_spatial_mobileclip2_v17 import HandSpatialMobileCLIP2V17Config, schema_fingerprint, schema_payload


LOG=logging.getLogger("hand_spatial_mobileclip2_v17")


def encode(path, visual, preprocess, device, config):
    crops,valid,boxes,crop_metadata=load_crop_archive(path)
    maps=np.zeros((16,3,512,8,8),dtype=np.float32);indices=np.argwhere(valid)
    if len(indices):
        tensors=torch.stack([preprocess(Image.fromarray(crops[t,v])) for t,v in indices]).to(device);chunks=[]
        with torch.inference_mode():
            for start in range(0,len(tensors),24):
                value=visual.trunk.stem(tensors[start:start+24])
                for stage in visual.trunk.stages:value=stage(value)
                chunks.append(value)
        encoded=torch.cat(chunks).float().cpu().numpy()
        if encoded.shape[1:]!=(512,8,8):raise RuntimeError(f"unexpected spatial shape {encoded.shape}")
        for item,(t,v) in enumerate(indices):maps[t,v]=encoded[item]
    if not np.isfinite(maps).all():raise RuntimeError(f"{path}: non-finite spatial maps")
    metadata={"schema_fingerprint":schema_fingerprint(config),"crop_schema_fingerprint":crop_metadata["schema_fingerprint"],"crop_archive":str(path),"video_path":crop_metadata["video_path"]}
    return maps.astype(np.float16),valid,boxes.astype(np.float16),metadata


def save(path,maps,valid,boxes,metadata,config):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+".tmp.npz")
    np.savez_compressed(temporary,spatial_maps=maps,valid=valid,boxes_normalized=boxes,metadata_json=np.array(json.dumps(metadata,sort_keys=True)),schema_json=np.array(json.dumps(schema_payload(config),sort_keys=True)));temporary.replace(path)


def run(args):
    if args.split not in ("train","val"):raise ValueError("the Citizen test split is sealed")
    config=HandSpatialMobileCLIP2V17Config();device=select_device(args.device);clip,preprocess=build_encoder(device);visual=clip.visual
    files=sorted((args.crop_root/args.split).glob("*/*.hand_rgb_v17.npz"));files=files[:args.limit] if args.limit else files
    started=time.monotonic();written=skipped=0;expected=schema_fingerprint(config)
    for index,path in enumerate(files,1):
        relative=path.relative_to(args.crop_root/args.split);stem=path.name.removesuffix(".hand_rgb_v17.npz");output=args.output_root/args.split/relative.parent/f"{stem}.hand_spatial_mobileclip2_v17.npz"
        if output.exists() and not args.overwrite:
            with np.load(output,allow_pickle=False) as payload:metadata=json.loads(str(payload["metadata_json"]))
            if metadata.get("schema_fingerprint")!=expected:raise ValueError(f"{output}: schema mismatch")
            skipped+=1;continue
        maps,valid,boxes,metadata=encode(path,visual,preprocess,device,config);save(output,maps,valid,boxes,metadata,config);written+=1
        if index==1 or index%10==0 or index==len(files):LOG.info("%s %d/%d written=%d skipped=%d elapsed=%.1fs",args.split,index,len(files),written,skipped,time.monotonic()-started)
    return {"split":args.split,"clips":len(files),"written":written,"skipped":skipped,"device":str(device),"schema_fingerprint":expected,"seconds":time.monotonic()-started,"test_accessed":False}


def build_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--split",required=True,choices=("train","val"));p.add_argument("--crop-root",type=Path,default=Path("data/local/citizen100_v17/hand_rgb"));p.add_argument("--output-root",type=Path,default=Path("data/local/citizen100_v17/hand_spatial_mobileclip2_s0"));p.add_argument("--device",default="auto");p.add_argument("--limit",type=int,default=0);p.add_argument("--overwrite",action="store_true");return p


def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s");print(json.dumps(run(build_parser().parse_args()),indent=2))


if __name__=="__main__":main()
