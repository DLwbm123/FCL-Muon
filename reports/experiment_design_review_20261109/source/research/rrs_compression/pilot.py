"""Only the RRS client-statistic transmission changes."""
import time
from research.adamw_t10.pilot import TenTaskPilot
from .codec import encode,decode


class CompressedPilot(TenTaskPilot):
    def aggregate_rrs_client(self,totals,statistics,client):
        cfg=self.cfg['rrs_compression']
        for name,statistic in statistics.items():
            self.budget();started=time.monotonic()
            packet=encode(statistic,rank=cfg['rank'],mode=cfg['mode'])
            reconstructed,record=decode(packet)
            # No raw-statistic addition or raw history exists on this path.
            totals[name]+=reconstructed
            self.charge('RRS_compressed_upload',len(packet))
            record.update(task=self.s['task'],client=client,layer=name,seconds=time.monotonic()-started)
            self.s.setdefault('rrs_compression_records',[]).append(record)
            self.event('rrs_compression',**record)
