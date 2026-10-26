"""Task-start normalized-feature preservation on current images only."""
import torch
from torch.nn import functional as F
from .head_loss import center_blocks
from .old_response import OldResponsePilot


def feature_anchor_mse(student,teacher,scale):
    if student.shape!=teacher.shape or student.ndim!=2 or student.shape[1]==0:
        raise ValueError('Feature targets must have identical nonempty BxD shape')
    return scale**2*F.mse_loss(student,teacher.detach())


def feature_preserving_backward(model,teacher,x,y,task,cpt):
    if task<1:raise ValueError('This bounded mechanism requires an old task')
    z=F.normalize(model.backbone.forward_features(x)[:,0],dim=-1)
    u=F.normalize(torch.cat(list(model.rows)),dim=-1)
    b_logits=model.scale*F.linear(z,u.detach())
    with torch.no_grad():target=F.normalize(teacher.backbone.forward_features(x)[:,0],dim=-1)
    ce=F.cross_entropy(b_logits[:,task*cpt:(task+1)*cpt],y-task*cpt)
    preservation=feature_anchor_mse(z,target,model.scale)
    b_loss=ce+preservation
    h_loss=F.cross_entropy(center_blocks(model.scale*F.linear(z.detach(),u),cpt),y)
    if not torch.isfinite(b_loss) or not torch.isfinite(h_loss):raise ArithmeticError('Nonfinite feature-preserving loss')
    b_loss.backward();h_loss.backward()
    return b_loss,h_loss,ce,preservation


class FeatureAnchorPilot(OldResponsePilot):
    def loss_terms(self,x,y):
        if self.cfg['feature_anchor_coefficient']!=1.:raise ValueError('Frozen feature coefficient changed')
        return feature_preserving_backward(self.model,self.teacher,x,y,self.s['task'],self.cfg['cpt'])
