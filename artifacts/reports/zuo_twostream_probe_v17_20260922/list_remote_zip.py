"""List the author's public checkpoint archive without downloading its payload."""
import io,json,urllib.request,zipfile
from pathlib import Path
class Remote(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
    def seekable(self):return True
    def seek(self,offset,whence=0):
        self.pos=offset+(self.pos if whence==1 else self.size if whence==2 else 0)
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def tell(self):return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        start=self.pos
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{start+n-1}'})
        with urllib.request.urlopen(req,timeout=60) as f:
            if f.status!=206 or not f.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                raise ValueError('server did not honor Range')
            b=f.read(n)
        assert len(b)==n
        self.pos+=len(b);return b
if __name__=='__main__':
    out=Path(__file__).parent
    for row in json.loads((out/'replacement_archives.json').read_text()):
        url=f"https://drive.usercontent.google.com/download?id={row['id']}&export=download&confirm=t"
        with zipfile.ZipFile(Remote(url,int(row['headers']['Content-Length']))) as z:
            files=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size,offset=i.header_offset) for i in z.infolist()]
        (out/(row['name']+'.inventory.json')).write_text(json.dumps(files,indent=2)+'\n')
        print(row['name'],len(files),'members',flush=True)
        print(json.dumps([f for f in files if any(k in f['name'].lower() for k in ['islr','online','phoenix','p14','s2g'])])[:6000],flush=True)
