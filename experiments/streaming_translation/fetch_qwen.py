"""Download only public pretrained model files; inference sends no data anywhere."""
from pathlib import Path
import requests

REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
DEST=Path(__file__).resolve().parent/'models/qwen05'

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','generation_config.json','model.safetensors','tokenizer.json',
                 'tokenizer_config.json','merges.txt','vocab.json','README.md']:
        target=DEST/name
        if target.exists():continue
        with requests.get(f'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REVISION}/{name}?download=true',
                          stream=True,timeout=30) as response:
            response.raise_for_status()
            temp=target.with_name(target.name+'.partial')
            with temp.open('wb') as f:
                for block in response.iter_content(4*1024*1024):f.write(block)
            if response.headers.get('Content-Length'):
                assert temp.stat().st_size==int(response.headers['Content-Length'])
            temp.rename(target)
        print(name,flush=True)
