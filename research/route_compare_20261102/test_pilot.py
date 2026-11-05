from research.route_compare_20261102.pilot import split_support_query


def test_client_split_retains_singletons_and_excludes_query_from_support():
    manifest={'labels':{'train':[0,0,1,2,3,3]},
              'partitions':{'train':[[[0,1,2],[],[3,4,5]]]}}
    support,query=split_support_query(manifest,0,3,4,76103)
    assert 2 in support[0] and 3 in support[2]
    assert support[1]==query[1]==[]
    for full,s,q in zip(manifest['partitions']['train'][0],support,query):
        assert set(s).isdisjoint(q) and sorted(s+q)==sorted(full)
    assert (support,query)==split_support_query(manifest,0,3,4,76103)


def test_shared_t1_fork_preserves_state_and_optimizer_identity(tmp_path):
    import copy
    import torch
    from fcl.model import QVAdapter
    from fcl.math import History
    from fcl.checkpoint import rng_state
    from research.aggregation_utility.lifecycle import import_prefix
    from research.route_compare_20261102.pilot import RoutePilot
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=QVAdapter(torch.nn.Linear(4,12),2)
            self.adapters={'0':self.a}
            self.rows=torch.nn.ParameterList()
        def add_head(self,c):self.rows.append(torch.nn.Parameter(torch.zeros(c,4)))
        def b_parameters(self):return [self.a.Bq,self.a.Bv]
    cfg={'cpt':2,'selection_mode':'prefix','output':'shared',
         'route':{'optimizer':'muon','mode':'prefix'}}
    clock={'started':1.,'deadline':2.};identity={'seed':0,'optimizer':'muon'}
    original=Model();original.add_head(2)
    history=History();history.commit(0,{'0':torch.eye(4,dtype=torch.float64)})
    source=RoutePilot(original,[None,None],{},None,None,None,cfg,tmp_path/'a',identity,clock)
    source.s.update(task=1,round=0,phase='task_start',merges=[0],history=history,
                    training_records=[{'visited':3}])
    ck={'schema':'cosine-head-regmean-pilot-1','state':copy.deepcopy(source.s),
        'config':copy.deepcopy(cfg),'identity':identity,'clock':clock,'head_count':1,
        'model':copy.deepcopy(original.state_dict()),'merged_flags':{'0':True},'rng':rng_state()}
    branch_cfg=copy.deepcopy(cfg);branch_cfg.update(selection_mode='selected',output='branch')
    branch_cfg['route']['mode']='selected'
    branch=RoutePilot(Model(),[None,None],{},None,None,None,branch_cfg,tmp_path/'b',identity,clock)
    fork=copy.deepcopy(ck);fork['config']['route']['mode']='selected'
    import_prefix(branch,fork)
    assert branch.s['history'].committed=={0}
    assert branch.s['prefix_provenance']['reused_training_visits']==3
    assert all(torch.equal(v,branch.model.state_dict()[k]) for k,v in ck['model'].items())
    assert ck['config']['route']['mode']=='prefix'
    wrong=copy.deepcopy(fork);wrong['config']['route']['optimizer']='adamw'
    import pytest
    with pytest.raises(ValueError,match='config mismatch'):
        import_prefix(RoutePilot(Model(),[None,None],{},None,None,None,branch_cfg,tmp_path/'c',identity,clock),wrong)
