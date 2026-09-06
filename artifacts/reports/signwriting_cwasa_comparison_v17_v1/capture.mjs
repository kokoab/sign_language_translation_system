// Isolated browser experiment; uses Node/Chrome already installed. No user profile.
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, writeFile, rm, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const base=new URL('.',import.meta.url).pathname, dir=join(base,process.argv[2]||'.');await mkdir(dir,{recursive:true});const profile=await mkdtemp(join(tmpdir(),'slt-cwasa-cdp-'));
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless','--no-first-run','--no-default-browser-check','--disable-background-networking','--remote-debugging-port=0','--user-data-dir='+profile,'--use-angle=swiftshader','--enable-unsafe-swiftshader','--window-size=1100,1000','about:blank'],{stdio:'ignore'});
let ws; const diagnostics=[]; const pending=new Map();let next=0;
try {
 let port; for(let i=0;i<40;i++){try{port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break}catch{await delay(250)}}
 if(!port)throw Error('Chrome debugging connection unavailable');
 const pages=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method && /Log.entryAdded|Runtime.exceptionThrown|Network.loadingFailed|Runtime.consoleAPICalled/.test(m.method))diagnostics.push(m);if(pending.has(m.id)){pending.get(m.id)(m);pending.delete(m.id)}};
 const call=async(method,params={})=>{
  const id=++next; const answer=new Promise(resolve=>pending.set(id,resolve));ws.send(JSON.stringify({id,method,params}));
  let timer;try{const m=await Promise.race([answer,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error(method+' timed out')),10000)})]);if(m.error)throw Error(JSON.stringify(m.error));return m.result}finally{clearTimeout(timer);pending.delete(id)}
 };
 await call('Page.enable');await call('Runtime.enable');await call('Log.enable');await call('Network.enable');
 // Deterministic capture clock for this headless process; engine bytes stay unchanged.
 await call('Page.addScriptToEvaluateOnNewDocument',{source:'window.requestAnimationFrame=cb=>setTimeout(()=>cb(performance.now()),16);window.cancelAnimationFrame=clearTimeout;window.sltTimerFrameFallback=true;'});
 await call('Page.navigate',{url:'http://127.0.0.1:8769/'+(process.argv[2]?process.argv[2]+'.html':'')+'?capture=1'});
 let value;
 for(let i=0;i<50;i++){
  await delay(500);
  const r=await call('Runtime.evaluate',{expression:'JSON.stringify({ready:document.readyState,timerFrameFallback:!!window.sltTimerFrameFallback,record:document.getElementById("frames")?.textContent,body:document.body?.innerText.slice(-2000)})',returnByValue:true});
  value=JSON.parse(r.result.value);
  if(i%10===0)console.log(JSON.stringify({ready:value.ready,recordBytes:value.record?.length,body:value.body?.slice(-200)}));
  const data=JSON.parse(value.record||'{}');
  if(data.paused && data.signs?.length && data.events.some(e=>e.type==='sigmlloaded'))break;
 }
 const data=JSON.parse(value.record||'{}');
 if(!data.paused || !data.signs?.length)throw Error('No paused signing animation generated');
 const evaluate=async expression=>(await call('Runtime.evaluate',{expression,returnByValue:true})).result.value;
 const total=data.signs.reduce((n,s)=>n+s.frames.length,0);
 const clip=JSON.parse(await evaluate('JSON.stringify((()=>{const r=document.querySelector("canvas").getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,scale:1}})())'));
 const captured=[];
 if(!process.argv.includes('--cas-only')){
  const frame=()=>evaluate('JSON.parse(document.getElementById("frames").textContent).lastFrame?.f');
  const step=async(direction,expected)=>{
   await evaluate(`document.querySelector(".bttn${direction}F.av0").click()`);
   for(let attempt=0;attempt<100;attempt++){await delay(20);if(await frame()===expected)return;}
   throw Error(`Frame step did not reach ${expected}; got ${await frame()}`);
  };
  let current=await frame();
  if(!Number.isInteger(current)) {await step('Next',1);current=1;}
  while(current!==0){const expected=(current+1)%total;await step('Next',expected);current=expected;}
  for(let i=0;i<total;i++){
   if(await frame()!==i)throw Error(`Capture frame mismatch: expected ${i}`);
   const shot=await call('Page.captureScreenshot',{format:'png',clip});
   await writeFile(join(dir,`step_${String(i).padStart(3,'0')}.png`),Buffer.from(shot.data,'base64'));
   captured.push({file:`step_${String(i).padStart(3,'0')}.png`,frame:i});
   if(i+1<total)await step('Next',i+1);
  }
 }
 await writeFile(join(dir,'capture_frames.json'),JSON.stringify(captured));
 console.log('generated',total,'frames; stepped capture',!process.argv.includes('--cas-only')); 
 await writeFile(join(dir,'cdp_diagnostics.json'),JSON.stringify(diagnostics));
 const extra=await call('Runtime.evaluate',{expression:'JSON.stringify({resources:performance.getEntriesByType("resource").map(r=>({url:r.name,duration:r.duration})),canvas:document.querySelector("canvas")?.outerHTML})',returnByValue:true});
 await writeFile(join(dir,'cdp_resources.json'),extra.result.value);
 await writeFile(join(dir,'cdp_capture.json'),JSON.stringify(value));
 const shot=await call('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,'cdp_browser.png'),Buffer.from(shot.data,'base64'));
 console.log('capture saved');
}finally{ws?.close();chrome.kill('SIGTERM');await delay(500);await rm(profile,{recursive:true,force:true,maxRetries:2})}
