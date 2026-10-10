from pathlib import Path
import subprocess,os,time,shutil
N=Path(__file__).resolve().parent;G=N.parent/'ordered_my_like_v17_20261011/gap175';R=N.parent/'my_like_regression_v17_20261011/resources';d=N/'legacy_native';d.mkdir(exist_ok=True)
for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','main.swift']:shutil.copy2(G/'native'/name,d/name)
p=d/'LiveReelDecoder.swift';s=p.read_text();needle='var c = LiveDecoderConfig(); c.maxInnerO = 8; c.innerFloor = 0.5; c.lag = 6; c.haveAngryCheck = true; c.logTheta = log(0.4); c.tiredRescueDroop = 1.0; return c';assert s.count(needle)==1
s=s.replace(needle,'var c = LiveDecoderConfig(); if ProcessInfo.processInfo.environment["PORT"] == "guards" { c.haveAngryCheck = true; c.tiredRescueDroop = 1.0 }; return c');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift']]
for kind,main in [('video',d/'main.swift'),('validation',G/'validation/main.swift')]:
 with (N/('legacy_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(N/('legacy_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
for arm,motion in [('base','off'),('guards','on')]:
 env=dict(os.environ,MODEL='SpanRecognizerV17LocalALettersB8FP16',LETTER_THRESHOLD='0.5',FINISH='1',MOTION=motion,PORT=arm)
 with (N/('legacy_'+arm+'_new_video.log')).open('w') as f:subprocess.run([str(N/'legacy_video_probe'),str(R),str(N/'new_video_cases.json'),str(N/('legacy_'+arm+'_new_video.json'))],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',arm,time.ctime(),flush=True)
