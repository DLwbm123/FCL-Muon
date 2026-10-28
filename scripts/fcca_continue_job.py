"""Same-clock continuation with separate receipts and whole-root storage budget."""
from scripts.fcca_job import main

if __name__=='__main__':
    main(config_name='fcca_ten_seed0.json',report_subdir='reports/ten',continuation=True)
