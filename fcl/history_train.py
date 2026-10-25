"""Research registry and lifecycle extension, reusing the existing FCIL trainer."""
import argparse,json,os
from pathlib import Path
from .model import MODELS,MuLoRAFL
from .history_optimizer import METHODS
from .history_extension import HistoricalCoordinator
from . import train


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',default=os.environ.get('RUN_CONFIG'));args,_=p.parse_known_args()
    if not args.config:
        train.main();return
    cfg=json.loads(Path(args.config).read_text());assert cfg['method'] in METHODS
    MODELS.update({name:MuLoRAFL for name in METHODS})
    train.main(HistoricalCoordinator(cfg))

if __name__=='__main__':main()
