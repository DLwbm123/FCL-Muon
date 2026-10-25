"""Parallel byte ranges of the same official archive, no mirror substitution."""
import concurrent.futures,json,os,time,urllib.request
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);dest=root/'data/cifar-100-python.tar.gz'
url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
size=169001437
parts=8

def download(i):
    low=i*size//parts;high=(i+1)*size//parts-1
    request=urllib.request.Request(url,headers={'Range':f'bytes={low}-{high}'})
    part=dest.with_suffix(f'.part{i}')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=60) as r:
                if r.status!=206:raise RuntimeError('Server ignored Range')
                if not r.headers.get('Content-Range','').startswith(f'bytes {low}-{high}/'):raise RuntimeError('Wrong byte range')
                with part.open('wb') as f:
                    total=0
                    while chunk:=r.read(1024*1024):
                        total+=len(chunk)
                        if total>high-low+1:raise RuntimeError('Range overflow')
                        f.write(chunk)
                assert total==high-low+1
            print('part',i,'complete',flush=True);return part
        except Exception:
            if attempt==2:raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:paths=list(pool.map(download,range(parts)))
tmp=dest.with_suffix('.complete')
with tmp.open('wb') as dst:
    for p in paths:
        with p.open('rb') as src:
            while chunk:=src.read(8*1024*1024):dst.write(chunk)
assert tmp.stat().st_size==size;tmp.replace(dest)
from torchvision.datasets import CIFAR100
CIFAR100(root/'data',train=True,download=True)
# Own temporary range copies only, after built-in archive check and extraction.
for p in paths:p.unlink()
print('CIFAR_READY',flush=True)
