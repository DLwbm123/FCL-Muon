"""Reuse the real-ViT synthetic exercise and audit its encoded statistics."""
import json
import os
from pathlib import Path
import torch
from research.adamw_t10.preflight import main as original
from research.adamw_t10.storage import write


def main():
    original()
    root=Path(os.environ['STUDY_ROOT'])
    checkpoint=torch.load(root/'src/runs/synthetic/latest.pt',map_location='cpu',weights_only=False)
    records=checkpoint['state']['rrs_compression_records']
    if len(records)!=24 or any(r['rank']!=64 for r in records):raise ValueError('missing production codec exercise')
    if 'RRS_raw_upload' in checkpoint['state']['communication']:raise ValueError('hidden dense upload')
    total=sum(r['wire_bytes'] for r in records)
    if total!=checkpoint['state']['communication']['RRS_compressed_upload']:raise ValueError('wire accounting mismatch')
    result=json.loads((root/'src/reports/preflight.json').read_text())
    result['RRS_codec']={'records':len(records),'actual_wire_bytes':total,'dense_reference_bytes':sum(r['dense_reference_tensor_bytes'] for r in records),
                         'max_diagonal_relative_error':max(r['relative_diagonal_error'] for r in records)}
    write(root,root/'src/reports/preflight.json',result)


if __name__=='__main__':main()
