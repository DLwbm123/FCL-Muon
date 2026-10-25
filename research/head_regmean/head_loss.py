"""Separated B/current-CE and head/current-or-all-seen-CE gradient paths."""
import torch
from torch.nn import functional as F
from .confirmation import ConfirmationPilot


def separated_backward(model,x,y,task,cpt,mode):
    if mode not in ('current','all_seen'):raise ValueError('Unknown head objective')
    z=F.normalize(model.backbone.forward_features(x)[:,0],dim=-1)
    u=F.normalize(torch.cat(list(model.rows)),dim=-1)
    b_logits=model.scale*F.linear(z,u.detach())
    b_loss=F.cross_entropy(b_logits[:,task*cpt:(task+1)*cpt],y-task*cpt)
    head_logits=model.scale*F.linear(z.detach(),u)
    h_loss=F.cross_entropy(head_logits,y) if mode=='all_seen' else F.cross_entropy(head_logits[:,task*cpt:(task+1)*cpt],y-task*cpt)
    if not torch.isfinite(b_loss) or not torch.isfinite(h_loss):raise ArithmeticError('Nonfinite separated CE')
    b_loss.backward();h_loss.backward()
    return b_loss,h_loss


class HeadLossPilot(ConfirmationPilot):
    expected_first=None
    first_batch_verified=False

    def backward_loss(self,x,y):
        b_loss,h_loss=separated_backward(self.model,x,y,self.s['task'],self.cfg['cpt'],self.cfg['head_loss_mode'])
        if self.expected_first is not None:
            expected=self.expected_first
            assert torch.equal(x.cpu(),expected['x']) and torch.equal(y.cpu(),expected['y'])
            assert all(torch.equal(p.grad.cpu(),g) for p,g in zip(self.model.b_parameters(),expected['B']))
            assert torch.equal(self.model.rows[-1].grad.cpu(),expected['head'])
            assert all(p.grad is None for p in self.model.rows[:-1])
            self.first_batch_verified=True;self.expected_first=None
            self.event('actual_first_batch_verified',head_loss_mode=self.cfg['head_loss_mode'],before_optimizer_step=True)
        return b_loss
