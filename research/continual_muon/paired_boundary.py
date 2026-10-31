"""Private capture and initialization helpers for a future paired diagnostic."""
import copy
import torch
from fcl.checkpoint import rng_state
from research.continual_muon.momentum_transport import candidate, select


def capture(pilot):
    state, model = pilot.s, pilot.model
    if (state['task'], state['round'], state['phase']) != (1, pilot.cfg['rounds'], 'ready_evaluation'):
        raise ValueError('not the end of T2 before state clear')
    names = list(model.adapters)
    if (set(state['client_states']) != set(range(pilot.cfg['clients'])) or
            any(a.merged for a in model.adapters.values())):
        raise ValueError('incomplete T2 client state or basis')
    momenta = {}
    for client, client_state in state['client_states'].items():
        values = client_state['m']
        if len(values) != 2 * len(names) or any(not torch.isfinite(m).all() for m in values):
            raise ValueError('invalid T2 momentum')
        momenta[client] = [m.detach().cpu().clone() for m in values]
    bases = [model.adapters[name].A.detach().cpu().clone() for name in names]
    return {'schema': 'paired-private-t2-boundary-v1', 'task': 1, 'round': state['round'],
            'identity': copy.deepcopy(pilot.identity), 'clock': copy.deepcopy(pilot.clock),
            'adapter_names': names, 'momenta': momenta, 'old_bases': bases, 'rng': rng_state()}


def install(pilot, snapshot, mode, probes=None):
    state, model = pilot.s, pilot.model
    if mode not in ('zero', 'projected', 'guarded'):
        raise ValueError('unknown paired arm')
    if (state['task'], state['round'], state['phase']) != (2, 0, 'round_start'):
        raise ValueError('not a fresh T3 task boundary')
    if (snapshot.get('schema') != 'paired-private-t2-boundary-v1' or snapshot.get('task') != 1 or
            snapshot.get('round') != pilot.cfg['rounds'] or snapshot.get('identity') != pilot.identity or
            snapshot.get('clock') != pilot.clock or snapshot.get('adapter_names') != list(model.adapters)):
        raise ValueError('private boundary identity mismatch')
    clients = set(range(pilot.cfg['clients']))
    if set(state['client_states']) != clients or set(snapshot['momenta']) != clients:
        raise ValueError('client set mismatch')
    if mode == 'guarded' and (probes is None or set(probes) != clients):
        raise ValueError('missing guarded training probes')
    if any(torch.count_nonzero(m) for v in state['client_states'].values() for m in v['m']):
        raise ValueError('T3 optimizer state is not initially zero')
    new_bases = [a.A.detach().cpu() for a in model.adapters.values()]
    staged, receipts = {}, {}
    for client in sorted(clients):
        projected, retention = candidate(snapshot['momenta'][client], snapshot['old_bases'], new_bases)
        current = state['client_states'][client]['m']
        if len(current) != len(projected) or any(a.shape != b.shape for a, b in zip(current, projected)):
            raise ValueError('T3 optimizer dimensions mismatch')
        if mode == 'zero':
            selected = [torch.zeros_like(v) for v in projected]
            info = {'accepted': False, 'retained_energy': retention, 'probe_cosine': None}
        elif mode == 'projected':
            selected = [v.clone() for v in projected]
            info = {'accepted': True, 'retained_energy': retention, 'probe_cosine': None}
        else:
            selected, info = select(projected, probes[client], retention)
        staged[client], receipts[client] = selected, info
    # No state changes occur if any validation or projection above fails.
    for client, values in staged.items():
        state['client_states'][client]['m'] = values
    return receipts
