"""Published HRNet whole-body extraction on one approved video frame, MPS."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'artifacts/vendor/zuo_runtime'))
import cv2,numpy as np,torch
from mmpose.apis import init_pose_model,inference_top_down_pose_model
import mmpose.apis.inference as api
from mmpose.datasets import DatasetInfo

def main():
    torch.set_num_threads(2)
    model=init_pose_model(str(ROOT/'artifacts/vendor/zuo_pose_configs/hrnet.py'),checkpoint=None,device='mps')
    ckpt=torch.load(ROOT/'artifacts/models/zuo_online_pretrained/hrnet_wholebody.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['state_dict'],strict=True);del ckpt
    # Legacy scatter assumes CUDA for all accelerators. Preserve metadata on CPU;
    # only transfer the prepared image tensor to MPS.
    original=api.scatter
    def scatter(inputs,devices):
        batches=original(inputs,[-1])
        for batch in batches:batch['img']=batch['img'].to('mps')
        return batches
    api.scatter=scatter
    rows=json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text())['records']
    row=next(r for r in rows if r['source']=='asllrp_contiguous' and r['role']=='validation')
    path=ROOT/row['video_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['video_sha256']
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,20);ok,frame=cap.read();cap.release();assert ok
    h,w=frame.shape[:2]
    # Full-frame single signer for this frontend smoke; final probe needs validated signer crop.
    person=[dict(bbox=np.array([0,0,w,h,1.],np.float32))]
    torch.mps.synchronize();start=time.perf_counter()
    try:
        results,_=inference_top_down_pose_model(model,frame,person,bbox_thr=None,format='xyxy',
            dataset='TopDownCocoWholeBodyDataset',dataset_info=DatasetInfo(model.cfg.dataset_info),return_heatmap=False)
    finally:api.scatter=original
    torch.mps.synchronize();elapsed=time.perf_counter()-start
    keypoints=results[0]['keypoints'];assert keypoints.shape==(133,3) and np.isfinite(keypoints).all()
    np.save(OUT/'pose_keypoints.npy',keypoints)
    selected=list(range(91,133))+list(range(71,91,2))+list(range(11))
    for x,y,c in keypoints[selected]:
        if c>.2:cv2.circle(frame,(round(float(x)),round(float(y))),3,(0,255,0),-1)
    cv2.imwrite(str(OUT/'pose_overlay.jpg'),frame)
    payload=dict(device='mps',strict_weights=True,source=row['video_path'],frame_index=20,shape=list(keypoints.shape),
                 seconds=elapsed,selected_channels=len(selected),confidence_mean=float(keypoints[selected,2].mean()),passed=True)
    (OUT/'pose_smoke.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
if __name__=='__main__':main()
