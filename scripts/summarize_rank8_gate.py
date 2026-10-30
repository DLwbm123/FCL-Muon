"""Reuse the matched all-seen audit with explicit uncompressed and cost gates."""
import summarize_all_seen_gate as audit
if __name__=='__main__':
    audit.OUT=audit.ROOT/'reports/rank8_gate_20261030'
    audit.REFERENCE_FOLDERS=dict(audit.REFERENCE_FOLDERS,uncompressed='all_seen_gate_20261030')
    audit.main()
