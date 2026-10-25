"""Load only the pinned LoRM dependency chain, without unrelated registry scans."""
import importlib
import sys
import types
from pathlib import Path


def load_lorm():
    root=Path(__file__).resolve().parents[1]/'sources/fed-mammoth'
    # Package-only bootstrap: original module files and algorithm methods unchanged.
    # The upstream __init__ files eagerly import all unrelated models and datasets.
    for package in ['_models','_networks','utils']:
        if package not in sys.modules:
            mod=types.ModuleType(package);mod.__path__=[str(root/package)]
            if package!='utils':
                setattr(mod,'register_model' if package=='_models' else 'register_network',lambda name:lambda cls:cls)
            sys.modules[package]=mod
    lorm=importlib.import_module('_models.lorm')
    vit=importlib.import_module('_networks.vit')
    return lorm.LoRM,vit.VisionTransformer
