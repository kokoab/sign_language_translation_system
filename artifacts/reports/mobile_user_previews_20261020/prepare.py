"""Normalize owner-supplied previews without changing framing or adding training inputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,subprocess
O=Path(__file__).resolve().parent
rows=json.loads((O/'inventory.json').read_text())['clips']
(O/'stage_by_id').mkdir(exist_ok=True)
def prepare(row):
    path=O/'stage_by_id'/(row['id']+'.mp4')
    if not path.exists():
        temporary=path.with_name(path.stem+'.partial.mp4')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',row['source'],
            '-map','0:v:0','-an','-vf','scale=720:720:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=30,setsar=1',
            '-c:v','libx264','-profile:v','main','-level:v','3.1','-crf','18','-preset','fast','-threads','2',
            '-pix_fmt','yuv420p','-movflags','+faststart','-map_metadata','-1',str(temporary)],check=True)
        temporary.replace(path)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:stream_side_data=rotation','-of','json',str(path)]))['streams'][0]
    assert probe['codec_name']=='h264' and max(probe['width'],probe['height'])<=720
    assert not any(s.get('rotation',0) for s in probe.get('side_data_list',[]))
    return {**row,'path':str(path.resolve()),'media':probe,'preview_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool:
    result=list(pool.map(prepare,rows))
(O/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'unnamed_videos.json').write_text(json.dumps([{'id':r['id'],'path':r['path']} for r in result if r['label'] is None],indent=2)+'\n')
print('Normalized',len(result),'clips; original input files unchanged.',flush=True)
