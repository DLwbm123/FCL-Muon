import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default='runs');a=p.parse_args()
for path in sorted(Path(a.root).glob('*/metrics.json')):
    m=json.loads(path.read_text());print(path.parent.name,json.dumps({k:v for k,v in m.items() if k not in ['matrix','stage_micro']}))
