"""Current-data old-response preservation for B; centered head learning stays separate."""
import torch
from torch.nn import functional as F
from .head_loss import HeadLossPilot,center_blocks


def old_response_mse(student,teacher,cpt):
    if student.shape!=teacher.shape or student.shape[1]==0 or student.shape[1]%cpt:
        raise ValueError('Old response shapes must contain complete nonempty task blocks')
    return F.mse_loss(center_blocks(student,cpt),center_blocks(teacher.detach(),cpt))


def response_preserving_backward(model,teacher,x,y,task,cpt):
    if task<1:raise ValueError('This bounded mechanism requires an old task')
    z=F.normalize(model.backbone.forward_features(x)[:,0],dim=-1)
    u=F.normalize(torch.cat(list(model.rows)),dim=-1)
    b_logits=model.scale*F.linear(z,u.detach())
    with torch.no_grad():target=teacher(x)[:,:task*cpt]
    ce=F.cross_entropy(b_logits[:,task*cpt:(task+1)*cpt],y-task*cpt)
    preservation=old_response_mse(b_logits[:,:task*cpt],target,cpt)
    b_loss=ce+preservation  # Fixed coefficient1 in scale16 logit units; no sweep.
    h_loss=F.cross_entropy(center_blocks(model.scale*F.linear(z.detach(),u),cpt),y)
    if not torch.isfinite(b_loss) or not torch.isfinite(h_loss):raise ArithmeticError('Nonfinite response-preserving loss')
    b_loss.backward();h_loss.backward()
    return b_loss,h_loss,ce,preservation


class OldResponsePilot(HeadLossPilot):
    teacher=None

    def loss_terms(self,x,y):
        if self.cfg['old_response_coefficient']!=1.:raise ValueError('Frozen protocol coefficient changed')
        return response_preserving_backward(self.model,self.teacher,x,y,self.s['task'],self.cfg['cpt'])

    def backward_loss(self,x,y):
        if self.teacher is None or any(v.requires_grad for v in self.teacher.parameters()):
            raise ValueError('A frozen task-start teacher is required')
        loss,head,ce,preserve=self.loss_terms(x,y)
        assert all(v.grad is None for v in self.teacher.parameters())
        assert all(v.grad is None for v in self.model.rows[:-1])
        if self.expected_first is not None:
            e=self.expected_first
            assert torch.equal(x.cpu(),e['x']) and torch.equal(y.cpu(),e['y'])
            assert all(torch.equal(v.grad.cpu(),g) for v,g in zip(self.model.b_parameters(),e['B']))
            assert torch.equal(self.model.rows[-1].grad.cpu(),e['head'])
            assert preserve.item()==0.
            self.first_batch_verified=True;self.expected_first=None
            self.event('actual_first_batch_verified',teacher_fixed=True,initial_preservation_loss_zero=True)
        r=self.s.setdefault('preservation_training',{'visits':0,'batches':0,'ce_sum':0.,'mse_sum':0.,'positive_mse_batches':0,'max_mse':0.})
        value=float(preserve.detach());r['visits']+=len(y);r['batches']+=1
        r['ce_sum']+=float(ce.detach())*len(y);r['mse_sum']+=value*len(y)
        r['positive_mse_batches']+=int(value>0);r['max_mse']=max(r['max_mse'],value)
        return loss
