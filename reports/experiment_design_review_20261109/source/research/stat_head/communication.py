"""Explicit reference distributed schedule, separate from measured wire bytes.

Forward dimensions follow each actual selected call. Backward is a conservative
unfused reference VJP schedule, not a claim of measured traffic or a lower bound.
Dense covariance adjoints are charged for the declared identity codec STE.
"""

def selection_ledger(routes,cfg,d=768):
    c=cfg['cpt'];clients=cfg['clients'];q=min(d,cfg['transport_rank']);rank=min(d,cfg['head_rank'])
    A=12*cfg['rank']*d*4;B=24*d*cfg['rank']*4;head=c*d*4
    codec=44+(d*rank+d)*4
    drift=8+4*(q+d+q*q+d*q)
    stats=drift+c*(8+d*4)+codec
    total={'forward_logical_reference_bytes':0,'backward_unfused_reference_bytes':0,'calls':0}
    rows=[]
    for route in routes:
        for call in route.get('calls',[]):
            rounds=call['head_fusions'];active=len(call['active_client_indices']) if 'active_client_indices' in call else sum(n>0 for n in route.get('split',{}).get('support_sizes',[1]*clients))
            C=(route['task']+1)*c
            formal=cfg['head_protocol'];w=C*(d+1)*4 if formal=='unified' else head
            forward=active*A+rounds*active*(2*(B+head)+(stats if formal=='unified' else codec))
            # Common-global statistics require post-aggregation B; the short
            # response cannot overlap its query/next round with this broadcast.
            if formal=='unified':forward+=rounds*active*B
            forward+=rounds*active*w+active*16  # head distribution, loss/count scalars
            backward=0
            if call['track_outer']:
                # Query gradients and reverse short rounds; unchanged A is
                # cached, but its adjoint must be reduced. Dense STE covariance
                # adjoints are explicit even though forward factors are small.
                query=active*(A+B+w)
                reverse=rounds*active*(2*(A+B+head)+d*d*8)
                if formal=='unified':reverse+=rounds*active*4*(q+d+q*q+d*q+c*d)
                backward=query+reverse
            rows.append(dict(task=route['task']+1,track_outer=call['track_outer'],
                forward_logical_reference_bytes=forward,backward_unfused_reference_bytes=backward))
            total['forward_logical_reference_bytes']+=forward;total['backward_unfused_reference_bytes']+=backward;total['calls']+=1
    return dict(scope='analytical reference distributed tensor schedule from actual selected calls; not measured network; no transport/retries/pretrained distribution',
        note='dense STE adjoints retained; fusion/caching or other valid distributed partitions may change these bytes',totals=total,rows=rows)
