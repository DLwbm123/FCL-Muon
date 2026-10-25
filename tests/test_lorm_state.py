import torch
from fcl.lorm_train import pack_server,restore_models
from types import SimpleNamespace


def test_lorm_round_snapshot_keeps_temporal_merge_and_client_counters():
    server=SimpleNamespace(network=torch.nn.Linear(4,3),cur_A={'w':torch.randn(2,4)},cur_B={'w':torch.randn(3,2)},
                           old_delta={'w':torch.randn(3,4)},run_weights_gram={'w':torch.randn(3,4)},
                           run_gram={'w':torch.eye(4)},optimization_dict={'w':torch.randn(3,4)},head={},
                           cur_task=1,cur_offset=10,cpt=10,cur_round=2,cur_train_matrix='A',is_server=True)
    snapshot=pack_server(server)
    other=SimpleNamespace(network=torch.nn.Linear(4,3))
    clients=[SimpleNamespace(network=torch.nn.Linear(4,3)) for _ in range(2)]
    counters=[dict(cur_task=1,cur_offset=10,cpt=10,cur_round=2,cur_train_matrix='A') for _ in clients]
    restore_models(other,clients,snapshot,counters)
    for name in ['cur_A','cur_B','old_delta','run_weights_gram','run_gram','optimization_dict']:
        torch.testing.assert_close(getattr(other,name)['w'],getattr(server,name)['w'])
    assert other.cur_train_matrix=='A' and other.cur_round==2
    for client in clients:
        assert client.old_delta is other.old_delta and client.cur_round==2
        torch.testing.assert_close(client.network.weight,server.network.weight)
