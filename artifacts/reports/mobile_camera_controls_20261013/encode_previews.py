"""Re-encode existing selected preview frames for iOS playback; preserve source selection."""
from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parents[3]
O=Path(__file__).resolve().parent
source=R/'ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples'
stage=O/'h264_previews';stage.mkdir(exist_ok=True)
backup=O/'before/media';backup.mkdir(parents=True,exist_ok=True)
def probe(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames,duration','-of','json',str(path)]))['streams'][0]
cat=json.loads((source/'catalog.json').read_text());rows=[]
for label,row in cat['examples'].items():
 name=row['assets']['loop'];src=source/name;dst=stage/name
 shutil.copy2(src,backup/name)
 before=probe(src)
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(src),'-map','0:v:0','-an','-c:v','libx264','-profile:v','main','-level:v','3.1','-crf','8','-pix_fmt','yuv420p','-fps_mode','passthrough','-movflags','+faststart',str(dst)],check=True)
 after=probe(dst)
 assert after['codec_name']=='h264' and after['pix_fmt']=='yuv420p'
 for key in ['width','height','nb_frames']:assert before[key]==after[key],(label,key,before,after)
 assert abs(float(before['duration'])-float(after['duration']))<0.001,(label,before,after)
 row['assets']['loop_sha256']=hashlib.sha256(dst.read_bytes()).hexdigest()
 rows.append({'gloss':label,'before':before,'after':after,'sha256':row['assets']['loop_sha256']})
shutil.copy2(source/'catalog.json',backup/'catalog.json')
cat['recognition_check']['preview_encoding_scope']='Original MPEG4 previews before H264 playback conversion; reference source selection unchanged. H264 previews not re-scored.'
cat['preview_encoding']={'codec':'H264 Main level 3.1','pixel_format':'yuv420p','crf':8,'faststart':True,'frames_dimensions_duration_preserved':True,'report':str(O.relative_to(R))}
(stage/'catalog.json').write_text(json.dumps(cat,indent=1)+'\n')
(O/'media_encoding.json').write_text(json.dumps({'clips':rows,'count':len(rows)},indent=2)+'\n')
print('Staged',len(rows),'H264 previews; originals preserved; not yet published')
