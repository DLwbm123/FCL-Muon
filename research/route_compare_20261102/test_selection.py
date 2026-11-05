import torch
from research.route_compare_20261102.frame import reference_frame
from research.route_compare_20261102.selection import select_basis


def test_layerwise_repair_retains_infeasible_layer_and_scores_whole_model():
    pool=torch.eye(3,dtype=torch.float64)
    reference=pool[:,:1].T.float()
    frame=reference_frame(pool,reference)
    frames={'free':frame,'blocked':frame}
    h={'free':torch.zeros(3,3,dtype=torch.float64),
       'blocked':torch.diag(torch.tensor([1.,3.,4.],dtype=torch.float64))}
    def evaluate(bases, track):
        loss=sum((a[0,1]-.25).square() for a in bases.values())
        return {'loss':loss,'current':loss*0}
    chosen,record=select_basis(frames,h,evaluate,steps=[.1],lambdas=[0.,1.],trust_radius=2.,
                              orthogonal_tolerance=1e-5,loss_tolerance=0.,space_tolerance=1e-8,
                              outer_iterations=1)
    assert record['changed_layers']==1
    assert record['reference_distances']['free']>1e-8
    assert record['reference_distances']['blocked']<1e-8
    assert record['selected_loss']<record['reference_loss']
    assert record['history'][0]['reason']=='accepted'
    assert torch.equal(chosen['blocked'],reference)


def test_failed_candidate_pool_freezes_only_that_layer():
    pool=torch.eye(3,dtype=torch.float64)
    reference=pool[:,:1].T.float()
    frame=reference_frame(pool,reference)
    frames={'free':frame,'failed':frame}
    histories={k:torch.zeros(3,3,dtype=torch.float64) for k in frames}
    def evaluate(bases, track):
        loss=sum((a[0,1]-.25).square() for a in bases.values())
        return {'loss':loss,'current':loss*0}
    selected,record=select_basis(frames,histories,evaluate,steps=[.1],lambdas=[0.],
        trust_radius=2.,orthogonal_tolerance=1e-5,loss_tolerance=0.,space_tolerance=1e-8,
        outer_iterations=1,frozen_layers={'failed'})
    assert torch.equal(selected['failed'],reference)
    assert record['changed_layers']==1
    assert record['history'][0]['layers']['failed']['reason']=='candidate_pool_failure'
