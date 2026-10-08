"""Merge results_*.json into one table: tune WER, Citizen val live view, user takes (GO, BIG+LANGUAGE)."""
import glob,json
rows={}
for f in sorted(glob.glob('results_*.json')):
    for k,v in json.load(open(f)).items():
        s,name=k.split(':',1);rows.setdefault(name,{})[s]=v['summary']
def fmt(name,r):
    t=r.get('tune',{});c=r.get('citizen_val',{});k=r.get('takes',{})
    go=sum(k[x]['GO'] for x in k if x in('01','02','03')) if k else None
    bl=sum(k[x]['BIG']+k[x]['LANGUAGE'] for x in ('01','02') if x in k) if k else None
    return (f"{name:22s} tune WER {t.get('wer','-')!s:6} S{t.get('S','-')} D{t.get('D','-')} I{t.get('I','-')} p90 {t.get('lat_p90','-')} | "
            f"citizen correct {c.get('correct','-')} exact {c.get('exact','-')} silent {c.get('silent','-')} extra {c.get('extra','-')} | "
            f"takes GO {go} BIG+LANG(01,02) {bl} words {sum(k[x]['n'] for x in ('01','02','03') if x in k) if k else '-'}")
for n,r in rows.items():print(fmt(n,r))
json.dump(rows,open('table.json','w'),indent=1)
