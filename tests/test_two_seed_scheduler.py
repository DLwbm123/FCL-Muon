import subprocess,sys,time
from scripts.ten_task_two_seed_job import identity,alive,next_seed_one_method


def test_additional_seed_does_not_wait_for_seed_zero_and_never_adds_seed_two():
    statuses={('regmean',0):{'status':'training'},('lorm',0):{'status':'training'}}
    assert next_seed_one_method(statuses)=='regmean'
    statuses[('regmean',1)]={'status':'training'}
    assert next_seed_one_method(statuses) is None
    statuses[('regmean',1)]={'status':'completed','qualified':True}
    assert next_seed_one_method(statuses)=='lorm'
    statuses[('lorm',1)]={'status':'completed','qualified':True}
    assert next_seed_one_method(statuses) is None


def test_qualification_or_technical_failure_blocks_future_launch():
    assert next_seed_one_method({('regmean',1):{'status':'completed','qualified':False}}) is None
    assert next_seed_one_method({('regmean',0):{'status':'failed'}}) is None


def test_identity_detects_exited_worker_and_rejects_pid_reuse():
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        saved=identity(p.pid);assert saved['pgid']==p.pid and alive(saved)
        assert not alive({**saved,'start_ticks':saved['start_ticks']+1})
        p.terminate();p.wait();assert not alive(saved)
    finally:
        if p.poll() is None:p.kill();p.wait()
