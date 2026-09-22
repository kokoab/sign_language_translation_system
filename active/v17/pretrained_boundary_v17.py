"""Bounded-window ASL adaptation of the pinned pretrained DGS pose encoder."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'artifacts/reports/pose_boundary_transfer_v17_20260922'
sys.path.insert(0,str(UPSTREAM/'dependencies'))
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_anonymization.data.normalization import normalize_mean_std
from safetensors.torch import load_file

FPS=20
FRAMES=64
LOOKAHEAD=10
TARGET=FRAMES-LOOKAHEAD-1
WEIGHTS=ROOT/'artifacts/models/pose_boundary_dgs_2026'


def should_stop(epoch,best_epoch):
    return epoch>=120 or (epoch>=40 and epoch-best_epoch>=20)


def window_indices(target,length):
    if not 0<=target<length:raise ValueError('target outside stream')
    idx=np.arange(target-TARGET,target+LOOKAHEAD+1)
    valid=(idx>=0)&(idx<length)
    return np.clip(idx,0,length-1),valid


def load_pose(row):
    with (ROOT/row['pose_path']).open('rb') as f:pose=Pose.read(f)
    duration=(len(pose.body.data)-1)/row['pose_fps']
    times=np.arange(int(np.floor(duration*FPS+1e-7))+1)/FPS
    indices=np.floor(times*row['pose_fps']+1e-7).astype(int)
    # Past observations only: zero-order hold on a20Hz source clock, never interpolation.
    return Pose(pose.header,NumPyPoseBody(FPS,pose.body.data[indices].copy(),pose.body.confidence[indices].copy()))


def observation_indices(valid,rng,sensor_fps=20.,dropout=0.):
    """Simulate held past sensor observations without moving the source clock."""
    if not 0<sensor_fps<=FPS or not 0<=dropout<1:raise ValueError('invalid observation augmentation')
    selected=np.arange(len(valid));positions=np.flatnonzero(valid)
    if not len(positions):raise ValueError('window has no observations')
    relative=np.arange(len(positions))
    ticks=np.floor(relative*sensor_fps/FPS+1e-9).astype(int)
    last=positions[0]
    for j,pos in enumerate(positions):
        if j and ticks[j]!=ticks[j-1] and rng.random()>=dropout:last=pos
        selected[pos]=last
    return selected


def window_features(pose,target,*,augmentation_seed=None):
    idx,valid=window_indices(target,len(pose.body.data))
    raw=pose.body.data[idx].filled(0).copy()
    confidence=pose.body.confidence[idx].copy()
    confidence[~valid]=0
    if augmentation_seed is not None:
        rng=np.random.default_rng(augmentation_seed)
        selected=observation_indices(valid,rng,sensor_fps=rng.uniform(15,20),dropout=rng.uniform(0,.15))
        raw=raw[selected];confidence=confidence[selected]
    window=Pose(pose.header,NumPyPoseBody(FPS,raw,confidence))
    window=normalize_mean_std(window)
    xyz=window.body.data.filled(0)[:,0].astype(np.float32)
    velocity=np.zeros_like(xyz)
    velocity[1:]=np.diff(xyz,axis=0)*FPS
    velocity[~valid]=0
    velocity[np.flatnonzero(valid)[0]]=0
    features=np.concatenate((xyz,velocity),axis=-1)
    features[~valid]=0
    if not np.isfinite(features).all():raise ValueError('nonfinite bounded-window normalization')
    times=(np.arange(FRAMES)-TARGET).astype(np.float32)/FPS
    return features,times


def backbone():
    path=UPSTREAM/'check.py'
    spec=importlib.util.spec_from_file_location('slt_pinned_pose_architecture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.model_from_upstream(json.loads((WEIGHTS/'config.json').read_text())).float()
    model.load_state_dict(load_file(str(WEIGHTS/'model.safetensors')),strict=True)
    return model


class PretrainedBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=backbone()
        self.edge=nn.Linear(384,2)
        self.set_adaptation(False)

    def set_adaptation(self,adapt):
        self.backbone.requires_grad_(False)
        if adapt:self.backbone.encoder_attn.requires_grad_(True)
        self.edge.requires_grad_(True)
        self.backbone.eval()
        self.backbone.encoder_attn.train(adapt and self.training)
        self.adapt=adapt

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()  # Freeze CNN batch-normalization statistics in both phases.
        self.backbone.encoder_attn.train(mode and getattr(self,'adapt',False))
        return self

    def project(self,x):
        return self.backbone.input_norm(self.backbone.frame_cnn(x))

    def encode_projected(self,x,times):
        for layer in self.backbone.encoder_attn:x=layer(x,times)
        return x[:,TARGET]

    def forward_projected(self,x,times):
        return self.edge(self.encode_projected(x,times))

    def forward(self,x,times):
        return self.forward_projected(self.project(x),times)
