"""Published Zuo online checkpoint: synthetic-input MPS compatibility, NOT accuracy."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'artifacts/vendor/slrt/Online/CSLR'))
import torch
import logging
import utils.misc
utils.misc.logger=logging.getLogger("zuo_probe")
import torch.nn.functional as F
from modelling.S3D import S3D_backbone
from modelling.two_stream import S3D_two_stream_v2
from modelling.Visualhead import SepConvVisualHead

def load_model():
    torch.set_num_threads(2)
    weights=ROOT/'artifacts/models/zuo_online_pretrained'
    vocab=json.loads((weights/'phoenix_vocab.json').read_text())
    assert len(vocab)==1116 and vocab[0]=='<blank>'
    expected=json.loads((OUT/'weights.json').read_text())['files']['phoenix_online.ckpt']['sha256']
    assert hashlib.sha256((weights/'phoenix_online.ckpt').read_bytes()).hexdigest()==expected
    # Skip only redundant K400 initialization. Strict complete online state replaces ALL tensors below.
    initialize=S3D_backbone.load_s3d_model_weight
    S3D_backbone.load_s3d_model_weight=lambda self,path:None
    model=torch.nn.Module()
    try:
        model.visual_backbone_twostream=S3D_two_stream_v2(use_block=5,freeze_block=(0,0),pose_inchannels=63,
            flag_lateral=(True,True),cfg_pyramid={'version':'v1','rgb':None,'pose':None},fusion_features=['c1','c2','c3','c4'])
    finally:S3D_backbone.load_s3d_model_weight=initialize
    for name,size in [('visual_head',1024),('visual_head_keypoint',1024),('visual_head_fuse',2048)]:
        setattr(model,name,SepConvVisualHead(cls_num=len(vocab),input_size=size,contras_setting=None,temp=.1))
    checkpoint=torch.load(weights/'phoenix_online.ckpt',map_location='cpu',weights_only=False)
    state={k.removeprefix('recognition_network.'):v for k,v in checkpoint['model_state'].items()}
    model.load_state_dict(state,strict=True)
    tensors=len(state)
    del state,checkpoint
    model.eval().to('mps')
    return model,vocab,tensors


def enable_mps_pooling(model):
    for layer in model.modules():
        if isinstance(layer,torch.nn.MaxPool3d):
            original=layer.forward
            layer.forward=lambda x,operation=original:operation(x.cpu()).to(x.device)
    original_avg=F.avg_pool3d
    F.avg_pool3d=lambda x,*a,**k:original_avg(x.cpu(),*a,**k).to(x.device) if x.device.type=='mps' else original_avg(x,*a,**k)
    return original_avg

def predict(model,rgb,pose):
    result=model.visual_backbone_twostream(rgb,pose)
    a=result['rgb_fea_lst'][-1];b=result['pose_fea_lst'][-1]
    logits=[model.visual_head(a)['gloss_logits'],model.visual_head_keypoint(b)['gloss_logits'],
            model.visual_head_fuse(torch.cat([a,b],dim=-1))['gloss_logits']]
    return torch.stack([x.softmax(-1) for x in logits]).mean(0)

def main():
    model,vocab,tensors=load_model()
    original_avg=enable_mps_pooling(model)
    rgb=torch.zeros(1,3,16,224,224,device='mps')
    pose=torch.zeros(1,63,16,112,112,device='mps')
    try:
        torch.mps.synchronize();start=time.perf_counter()
        with torch.inference_mode():actual=predict(model,rgb,pose).cpu()
        torch.mps.synchronize();elapsed=time.perf_counter()-start
        assert actual.shape==(1,1116) and torch.isfinite(actual).all()
        model.cpu();rgb=rgb.cpu();pose=pose.cpu()
        with torch.inference_mode():expected=predict(model,rgb,pose)
        torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-3)
    finally:F.avg_pool3d=original_avg
    result=dict(strict_tensors=tensors,device='mps',cpu_fallback='3D pooling',synthetic_input=True,
                input_rgb=[1,3,16,224,224],input_pose=[1,63,16,112,112],output=list(actual.shape),
                inference_seconds=elapsed,cpu_mps_max_abs_difference=float((actual-expected).abs().max()),
                passed=True,accuracy_tested=False)
    (OUT/'model_smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
