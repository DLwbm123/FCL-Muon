import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args()
for name in a.paths:
 m=json.loads(Path(name).read_text());cpt=m['cpt']
 for split in ['train','test']:
  all_ids=[]
  for t,clients in enumerate(m['partitions'][split]):
   for ids in clients:
    if split=='train':assert len(ids)>=6
    assert all(t*cpt<=m['labels'][split][i]<(t+1)*cpt for i in ids)
    all_ids+=ids
  assert sorted(all_ids)==list(range(len(m['labels'][split])))
 print(name,'VALID')
