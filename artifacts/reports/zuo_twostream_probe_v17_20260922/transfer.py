"""Zero-shot blank transfer on 20 pre-existing held-out interval centres, no fit."""
import json,sys,time,hashlib,contextlib
from pathlib import Path
import model_smoke as online
ROOT=online.ROOT;OUT=online.OUT
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'artifacts/vendor/zuo_runtime'))
import cv2,numpy as np,torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from mmpose.apis import init_pose_model,inference_top_down_pose_model
from mmpose.datasets import DatasetInfo
import mmpose.apis.inference as api
from utils.gen_gaussian import gen_gaussian_hmap_op


def letterbox(frame,points):
    h,w=frame.shape[:2];scale=min(224/w,224/h);nw,nh=round(w*scale),round(h*scale)
    left,top=(224-nw)//2,(224-nh)//2
    image=np.zeros((224,224,3),np.uint8)
    image[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh))
    points=points.copy();points[:,0]=points[:,0]*(nw/w)+left;points[:,1]=points[:,1]*(nh/h)+top
    return image,points


def main():
    torch.set_num_threads(2)
    from active.v17.approved_phrase_data_v17 import verify_manifest
    verify_manifest()
    recipe=json.loads((OUT/'transfer_recipe.json').read_text())
    for path,sha in recipe['inputs'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    model,vocab,_=online.load_model();original_avg=online.enable_mps_pooling(model)
    pose=init_pose_model(str(ROOT/'artifacts/vendor/zuo_pose_configs/hrnet.py'),checkpoint=None,device='mps')
    ckpt=torch.load(ROOT/'artifacts/models/zuo_online_pretrained/hrnet_wholebody.pth',map_location='cpu',weights_only=False)
    pose.load_state_dict(ckpt['state_dict'],strict=True);del ckpt
    info=DatasetInfo(pose.cfg.dataset_info)
    original_scatter=api.scatter
    def scatter(inputs,devices):
        batches=original_scatter(inputs,[-1])
        for batch in batches:batch['img']=batch['img'].to('mps')
        return batches
    api.scatter=scatter
    evidence=json.loads((ROOT/'artifacts/reports/reel_decision_probe_v17_20260922/evidence.json').read_text())['results']
    rows=[r for r in evidence if r['role']=='validation' and r['kind'] in ['core','gap']]
    assert len(rows)==20 and sum(r['kind']=='gap' for r in rows)==3
    manifest=json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text())['records']
    hashes={r['video_path']:r['video_sha256'] for r in manifest}
    indices=list(range(91,133))+list(range(71,91,2))+list(range(11))
    results=[];start=time.perf_counter();pose_total=0.;model_total=0.
    try:
        for video in sorted({r['video'] for r in rows}):
            assert hashlib.sha256((ROOT/video).read_bytes()).hexdigest()==hashes[video]
            cap=cv2.VideoCapture(str(ROOT/video));fps=cap.get(cv2.CAP_PROP_FPS);source_n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            cache=ROOT/'artifacts/reports/shubert_decision_probe_v17_20260922/cache'/hashlib.sha256(video.encode()).hexdigest()[:16]
            cap=cv2.VideoCapture(str(cache/'signer_crop.mp4'));frames=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                frames.append(frame)
            cap.release();assert len(frames)==source_n
            prepared={}
            for row in [r for r in rows if r['video']==video]:
                centre=(row['start']+row['end'])/2
                clocks=centre+(np.arange(16)-7)/25
                selected=np.clip(np.rint(clocks*fps).astype(int),0,len(frames)-1)
                for idx in sorted(set(selected)-set(prepared)):
                    frame=frames[idx];h,w=frame.shape[:2];t=time.perf_counter()
                    predictions,_=inference_top_down_pose_model(pose,frame,[dict(bbox=np.array([0,0,w,h,1.],np.float32))],
                        format='xyxy',dataset='TopDownCocoWholeBodyDataset',dataset_info=info)
                    torch.mps.synchronize();pose_total+=time.perf_counter()-t
                    kp=predictions[0]['keypoints'];assert kp.shape==(133,3) and np.isfinite(kp).all()
                    prepared[idx]=letterbox(frame,kp[indices])
                images=np.stack([prepared[i][0] for i in selected]);points=np.stack([prepared[i][1] for i in selected])
                rgb=torch.from_numpy(images).to('mps',dtype=torch.float32).permute(3,0,1,2).unsqueeze(0)/127.5-1
                coords=torch.from_numpy(points).to('mps')
                heat=torch.cat([resize(gen_gaussian_hmap_op(chunk,raw_size=(224,224),sigma=8),[112,112],antialias=True) for chunk in coords.split(4)])
                heat=(heat.permute(1,0,2,3).unsqueeze(0)-.5)/.5
                torch.mps.synchronize();t=time.perf_counter()
                with torch.inference_mode():prob=online.predict(model,rgb,heat)[0].cpu().numpy()
                torch.mps.synchronize();model_total+=time.perf_counter()-t
                assert prob.shape==(1116,) and np.isfinite(prob).all() and abs(float(prob.sum())-1)<1e-4
                predicted=int(prob.argmax());passes=predicted!=0
                results.append(dict(id=row['id'],video=video,kind=row['kind'],target=row['target'],blank_probability=float(prob[0]),
                    checkpoint_top_gloss=vocab[predicted],pass_gate=passes,source_indices=selected.tolist(),
                    baseline_commit=row['conditional_commit'],baseline_correct=bool(row['conditional_commit'] and row['target']==row['verifier']['candidate_gloss']),
                    candidate=row['verifier']['candidate_gloss'],mean_pose_confidence=float(points[:,:,2].mean())))
                print(json.dumps(results[-1]),flush=True)
    finally:api.scatter=original_scatter;F.avg_pool3d=original_avg
    gaps=[r for r in results if r['kind']=='gap'];cores=[r for r in results if r['kind']=='core']
    summary=dict(core_count=len(cores),gap_count=len(gaps),gaps_rejected=sum(not r['pass_gate'] for r in gaps),
        baseline_false_gap_commits=sum(r['baseline_commit'] for r in gaps),false_gap_commits=sum(r['baseline_commit'] and r['pass_gate'] for r in gaps),
        baseline_correct_core_commits=sum(r['baseline_correct'] for r in cores),correct_core_commits=sum(r['baseline_correct'] and r['pass_gate'] for r in cores),
        core_gate_pass=sum(r['pass_gate'] for r in cores),wall_seconds=time.perf_counter()-start,pose_seconds=pose_total,recognizer_seconds=model_total)
    payload=dict(summary=summary,rows=results,recipe_sha256=hashlib.sha256((OUT/'transfer_recipe.json').read_bytes()).hexdigest(),
        limitations=['Zero-shot German-trained blank head on ASL, no 100-sign/WER result','Oracle interval centres with fixed16frame windows at25Hz, not full live scheduler',
                     'Aspect-preserving224square letterbox differs from upstream direct square resize; heatmaprawsize224square,sigma8',
                     'HRNetW48DARK public checkpoint; exact author pose checkpoint identity unconfirmed','Three reused-development gaps, no threshold tuning or head fit'])
    (OUT/'transfer_results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
