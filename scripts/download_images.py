"""Optional official ImageNet-R download; identities still come from pinned YAML."""
import json,os,tarfile,time,urllib.request
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);data=root/'data';data.mkdir(exist_ok=True)
url='https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
archive=data/'imagenet-r.tar'
started=time.time();limit=5_000_000_000
with urllib.request.urlopen(url,timeout=60) as src,archive.open('wb') as dst:
    size=0
    while chunk:=src.read(8*1024*1024):
        size+=len(chunk)
        if size>limit or time.time()-started>84000:raise RuntimeError('Download size/time guard')
        dst.write(chunk)
with tarfile.open(archive) as tar:tar.extractall(data,filter='data')
(root/'reports/imagenetr_download.json').write_text(json.dumps({'source':url,'bytes':size,'end':time.time()}))
print('IMAGENETR_IMAGES_READY',flush=True)
