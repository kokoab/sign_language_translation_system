"""Fetch only the author's online PHOENIX checkpoint and vocabulary via HTTP ranges."""
import hashlib,json,zipfile
from pathlib import Path
from list_remote_zip import Remote
out=Path(__file__).parent
models=out.parents[1]/'models/zuo_online_pretrained'
models.mkdir(parents=True,exist_ok=True)
row=next(r for r in json.loads((out/'replacement_archives.json').read_text()) if r['name']=='EMNLP24_Online.zip')
url=f"https://drive.usercontent.google.com/download?id={row['id']}&export=download&confirm=t"
files={'EMNLP24_Online/Phoenix-2014T/ckpts/cslr_best.ckpt':'phoenix_online.ckpt',
       'EMNLP24_Online/Phoenix-2014T/meta/phoenix_iso_with_blank.vocab':'phoenix_vocab.json'}
result={}
with zipfile.ZipFile(Remote(url,int(row['headers']['Content-Length']))) as z:
    for member,name in files.items():
        data=z.read(member) # zipfile verifies CRC; only these two explicitly selected assets.
        (models/name).write_bytes(data)
        result[name]=dict(member=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        print(name,len(data),flush=True)
(out/'weights.json').write_text(json.dumps(dict(archive_url=url,files=result),indent=2)+'\n')
