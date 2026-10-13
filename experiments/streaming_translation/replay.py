"""Local, causal recognition-event replay. No production imports or writes."""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROMPT = ('Translate these ASL glosses into natural English. Preserve who does what, '
          'ownership, time and negation. Do not invent missing information. '
          'An incomplete input may remain incomplete. Output only the English translation.\nGlosses: ')
FEWSHOT = ('Convert ASL glosses to English. Do not answer questions or invent information.\n'
           'ASL: WE HAPPY\nEnglish: We are happy.\n'
           'ASL: YOUR NAME WHAT\nEnglish: What is your name?\n'
           'ASL: I NO UNDERSTAND\nEnglish: I do not understand.\n'
           'ASL: TOMORROW WE GO HOME\nEnglish: We will go home tomorrow.\n'
           'ASL: {glosses}\nEnglish:')

def norm(text):
    return ' '.join(re.findall(r"[a-z0-9]+", text.lower().replace("don't", 'do not')))

def sentences(text):
    return re.findall(r'[^.!?]+[.!?]', text)

def common_sentences(a, b):
    result = []
    for x, y in zip(sentences(a), sentences(b)):
        if x.strip() != y.strip(): break
        result.append(x.strip())
    return ' '.join(result)

def erasure(a, b):
    x, y = a.split(), b.split()
    n = 0
    for u, v in zip(x, y):
        if u != v: break
        n += 1
    return len(x)-n

class Translator:
    def __init__(self, name):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
        torch.set_num_threads(4)
        self.torch, self.name = torch, name
        path = str(ROOT/'artifacts/models/stage3_multisentence_tiny_v17_20260929') if name=='tiny' else 'google/flan-t5-base'
        if name.startswith('qwen'): path=str(HERE/'models/qwen05')
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls=AutoModelForCausalLM if name.startswith('qwen') else AutoModelForSeq2SeqLM
        self.model = cls.from_pretrained(path, local_files_only=True).eval()
        self.path = HERE / f'{name}_generations.jsonl'
        self.cache = {}
        if self.path.exists():
            for line in self.path.open():
                row = json.loads(line); self.cache[row['key']] = row
        self.file = self.path.open('a', buffering=1)
        # Warm-up uses neither experimental cases nor references.
        x = self.tokenizer('hi yes' if name=='tiny' else PROMPT+'YES', return_tensors='pt')
        with torch.inference_mode(): self.model.generate(**x, max_new_tokens=10, do_sample=False)

    def __call__(self, glosses, scores):
        buckets = ['hi' if s >= .6 else 'mid' if s >= .4 else 'lo' for s in scores]
        key = json.dumps([glosses,buckets])
        if key in self.cache: return self.cache[key]
        slots, inputs = {}, []
        for gloss in glosses:
            if gloss.startswith('fs-'):
                slot = f'FS{len(slots)}'; slots[slot] = gloss[3:].title(); inputs.append(slot)
            else: inputs.append(gloss)
        source = ' '.join(f'{b} {g.lower()}' for b,g in zip(buckets,inputs)) if self.name=='tiny' else PROMPT+' '.join(inputs)
        if self.name in ('flan_fewshot','qwen'):
            source=FEWSHOT.format(glosses=' '.join(inputs))
        if self.name=='qwen':
            source=self.tokenizer.apply_chat_template([{'role':'user','content':source}],tokenize=False,add_generation_prompt=True)
        if self.name=='qwen_direct':
            messages=[{'role':'system','content':
                'Translate the user ASL glosses into English. Output only the translation. '
                'Do not answer questions or invent names, events, people or tense. '
                'Preserve negation. Leave incomplete input unfinished.'},
                {'role':'user','content':' '.join(inputs)}]
            source=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        x = self.tokenizer(source, return_tensors='pt')
        start = time.perf_counter()
        mode = 'model'
        if self.name=='tiny' and x.input_ids.shape[1] > 64:
            raw = ' '.join(inputs).capitalize()+'.'; mode = 'context_literal_fallback'
        else:
            with self.torch.inference_mode():
                y = self.model.generate(**x, max_new_tokens=63 if self.name=='tiny' else 96,
                                        do_sample=False, num_beams=1)
            tokens=y[0][x.input_ids.shape[1]:] if self.name.startswith('qwen') else y[0]
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        ms = (time.perf_counter()-start)*1000
        if slots:
            if all(len(re.findall(r'\b'+k+r'\b', raw, re.I))==1 for k in slots):
                for k,v in slots.items(): raw = re.sub(r'\b'+k+r'\b',v,raw,flags=re.I)
            else:
                raw = ' '.join(slots.get(g,g.lower()) for g in inputs).capitalize()+'.'
                mode = 'slot_literal_fallback'
        parts = re.findall(r'(?:^|\s)(\d+):\s*(.*?)(?=\s+\d+:|$)',raw)
        text = ' '.join(t.strip() for _,t in parts) if parts else raw
        row = dict(key=key,glosses=glosses,buckets=buckets,source=source,raw=raw,text=text,
                   ms=ms,mode=mode,input_tokens=x.input_ids.shape[1])
        self.cache[key]=row
        self.file.write(json.dumps(row)+'\n')
        return row

def simulate(row, tr, policy, gap=1.5):
    """All decisions use the received prefix. Final reference is never an input."""
    gs, scores = row['glosses'],row['scores']
    ts = row['arrivals']
    if ts is None: return None
    ts = [t-ts[0] for t in ts]
    events=[]; committed=''; previous=''; displayed=''; busy=0.; cut=0
    erased=0; conflicts=0

    def render(a,b,at,kind):
        nonlocal busy, displayed, erased
        result=tr(gs[a:b],scores[a:b])
        begin=max(at,busy); busy=begin+result['ms']/1000
        text=result['text']
        events.append(dict(at=at,ready=busy,queue_delay=begin-at,kind=kind,n=b,
                           text=text,ms=result['ms'],mode=result['mode']))
        return text

    if policy=='finish':
        # Oracle receipt of final word is an optimistic proxy, NOT logged Finish time.
        displayed=render(0,len(gs),ts[-1],'oracle_end')
        committed=displayed
    elif policy=='pause':
        chunks=[]
        for i,t in enumerate(ts):
            if i and t-ts[i-1]>gap:
                chunks.append(render(cut,i,ts[i-1]+gap,'timeout'))
                cut=i
        chunks.append(render(cut,len(gs),ts[-1]+gap,'timeout'))
        displayed=committed=' '.join(chunks)
    else:
        for i,t in enumerate(ts):
            text=render(0,i+1,t,'draft')
            erased+=erasure(displayed,text)
            displayed=text
            if policy=='adaptive':
                stable=common_sentences(previous,text)
                # Frozen output and full source history are independent. A conflict
                # prevents further speaking; it does not silently revise spoken text.
                if committed and not text.startswith(committed): conflicts+=1
                elif stable.startswith(committed) and len(stable)>len(committed):
                    committed=stable
                    events.append(dict(at=t,ready=busy,kind='commit',n=i+1,text=committed))
            previous=text
        # No automatic end-of-stream flush: adaptive must earn a final commitment.
    full=tr(gs,scores)['text']
    finals=[e['ready'] for e in events if e['kind'] in ('commit','timeout','oracle_end')]
    return dict(id=row['id'],family=row['family'],policy=policy,final=displayed,
                committed=committed,full=full,full_equal=norm(displayed)==norm(full),
                tail_finalized=bool(committed) and norm(committed)==norm(displayed),
                conflicts=conflicts,erased_tokens=erased,
                normalized_erasure=erased/max(1,len(displayed.split())),
                first_caption_seconds=events[0]['ready'],
                last_word_seconds=ts[-1],last_caption_lag=busy-ts[-1],
                final_commit_lag=(finals[-1]-ts[-1]) if finals else None,
                max_queue_seconds=max((e.get('queue_delay',0) for e in events),default=0),events=events)

def selfcheck():
    assert common_sentences('Hello. I want','Hello. I need')=='Hello.'
    assert common_sentences('I want water.','I do not want water.')==''
    assert erasure('I want water','I want food')==1
    class Stub:
        def __call__(self,g,s): return dict(text=' '.join(g)+'.',ms=10,mode='stub')
    row=dict(id='x',family='test',glosses=['I','WANT','WATER'],scores=[1]*3,arrivals=[0,1,4])
    pause=simulate(row,Stub(),'pause')
    assert pause['events'][0]['n']==2 and pause['events'][0]['at']==2.5
    assert not simulate(row,Stub(),'adaptive')['tail_finalized']
    assert simulate(row,Stub(),'revisable')['full_equal']
    assert all(e['ready']>=e['at'] for e in pause['events'])
    later=dict(row,glosses=['I','WANT','FOOD'])
    first=simulate(row,Stub(),'revisable')['events'][:2]
    second=simulate(later,Stub(),'revisable')['events'][:2]
    assert first==second, 'Future words changed earlier captions'
    class WrongButStable:
        def __call__(self,g,s):
            return dict(text='He gives me water.' if len(g)<3 else 'I want water.',ms=10,mode='stub')
    failure=simulate(row,WrongButStable(),'adaptive')
    assert failure['conflicts']==1 and failure['committed']=='He gives me water.'
    print('selfcheck passed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['tiny','flan','flan_fewshot','qwen','qwen_direct','selfcheck'])
    args=parser.parse_args()
    if args.model=='selfcheck': selfcheck();return
    rows=json.loads((HERE/'cases.json').read_text());tr=Translator(args.model)
    results=[];outputs=[]
    for i,row in enumerate(rows):
        full=tr(row['glosses'],row['scores'])
        refs=row.get('references',[])
        outputs.append(dict(id=row['id'],family=row['family'],glosses=row['glosses'],
                            text=full['text'],ms=full['ms'],mode=full['mode'],references=refs,
                            reference_match=any(norm(full['text'])==norm(r) for r in refs) if refs else None))
        for policy in ['finish','pause','revisable','adaptive']:
            result=simulate(row,tr,policy)
            if result: results.append(result)
        if (i+1)%5==0: print(f'{args.model}: {i+1}/{len(rows)} cases, {len(tr.cache)} unique generations',flush=True)
    (HERE/f'{args.model}_outputs.json').write_text(json.dumps(outputs,indent=2)+'\n')
    (HERE/f'{args.model}_replay.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'DONE {args.model}',flush=True)

if __name__=='__main__': main()
