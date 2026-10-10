"""Small causal regression check for experimental identity/publication separation."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]));sys.path.insert(0,str(P))
import runtime_experiment as mod
from test.test_segmental_runtime_v17 import HaveContactRegression
rt=HaveContactRegression.runtime();rt.__class__=mod.SegmentalRuntime
rt.have_contact_closure=False;rt.count,rt.context,rt.lag=21,2,6
rt.config={'stream':{'min_span_frames':3,'max_inner_o':8,'experiment_arm':'ordered'}}
rt.early_k,rt.early_q,rt.early_min=3,.9,4;rt.early_unsafe={'HAVE'}
rt.early={11:('BIG',2)};rt.soft_extra,rt.hard,rt.letter_gap=2,.5,6
rt.last=None;rt.pending_output=[]
rt.obs=[SimpleNamespace(seconds=i/20.) for i in range(21)]
rt._score=lambda spans:None;rt._trim=lambda:None
segments=[dict(start=2,end=10,gloss='HAVE',log_q=np.log(.65)),dict(start=11,end=15,gloss='BIG',log_q=np.log(.99))]
with patch.object(mod,'both_spans',return_value=[(2,10),(11,15)]),patch.object(mod,'segmental_decode',return_value=segments):
 assert rt._step(final=False)==[], 'BIG must wait for HAVE publication'
 assert rt.last['gloss']=='BIG' and rt.shown_early[0][1] is rt.last, 'identity must lock without delay'
 assert len(rt.pending_output)==1
 words=rt._step(final=True)
 assert [w['gloss'] for w in words]==['HAVE','BIG']
 assert words[1]['closed'] and words[1]['commit_frame']==20
 assert rt.pending_output==[]
 assert rt._step(final=True)==[], 'flush must not publish twice'
rt.pending_output=[{'gloss':'BIG'}];rt.boundary=Mock();rt.reset();assert not rt.pending_output
print('PASS: early identity preserved, HAVE before BIG, final flush exactly once, reset clears pending output.')
import runtime_ends as ends
rt=HaveContactRegression.runtime();rt.__class__=ends.SegmentalRuntime
rt.have_contact_closure=False;rt.count,rt.context,rt.lag=21,2,6
rt.config={'stream':{'min_span_frames':3,'max_inner_o':8,'experiment_arm':'ordered'}}
rt.early_k=0;rt.soft_extra=0;rt.letter_gap=6;rt.pending_output=[]
rt.obs=[SimpleNamespace(seconds=i/20.) for i in range(21)]
rt._score=lambda spans:None;rt._trim=lambda:None
rt.last=rt._word('NAME',2,5,10,np.log(.99)); rt.early={}
segments=[dict(start=2,end=10,gloss='NAME',log_q=np.log(.99))]
with patch.object(ends,'both_spans',return_value=[(2,10)]),patch.object(ends,'segmental_decode',return_value=segments):
 assert rt._step(final=True)==[]
 assert rt.last['end_frame']==10 and rt.last['end_seconds']==.5, 'frame/time endpoints must agree'
print('PASS: repeated-word merge advances the frame and timestamp together.')
import runtime_ordered_v2 as v2
rt=HaveContactRegression.runtime();rt.__class__=v2.SegmentalRuntime
rt.have_contact_closure=False;rt.count,rt.context,rt.lag=76,2,6
rt.config={'stream':{'min_span_frames':3,'max_inner_o':8,'experiment_arm':'ordered'}}
rt.early_k,rt.early_q,rt.early_min=3,.9,4;rt.early_unsafe=set();rt.early={49:('HELP',2)}
rt.soft_extra=0;rt.letter_gap=6;rt.pending_output=[];rt.last=None
rt.obs=[SimpleNamespace(seconds=i/20.) for i in range(90)]
rt.bio=[rt.bio[0]]*70;rt._score=lambda spans:None;rt._trim=lambda:None
segments=[dict(start=16,end=38,gloss='PLEASE',log_q=np.log(.99)),dict(start=49,end=66,gloss='HELP',log_q=np.log(.99))]
with patch.object(v2,'both_spans',return_value=[(16,38),(49,66)]),patch.object(v2,'segmental_decode',return_value=segments):
 words=rt._step(final=False)
 assert [w['gloss'] for w in words]==['PLEASE','HELP']
 assert rt.last['gloss']=='HELP', 'an earlier PLEASE must not replace the later HELP in duplicate tracking'
origin=rt.origin;rt.bio=[rt.bio[0]]*80;rt.count=86
segments=[dict(start=63-origin,end=73-origin,gloss='HELP',log_q=np.log(.99))]
with patch.object(v2,'both_spans',return_value=[(63-origin,73-origin)]),patch.object(v2,'segmental_decode',return_value=segments):
 assert rt._step(final=True)==[], 'later HELP fragment must merge with the first HELP'
 assert rt.last['end_frame']==73 and rt.last['end_seconds']==73/20
print('PASS: earlier closure cannot corrupt latest-word identity; HELP does not duplicate.')
