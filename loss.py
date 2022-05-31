''' Adapted from https://github.com/ZeroE04/R-CenterNet/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()  
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4)
  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
  num_pos  = pos_inds.float().sum() 
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos 
  return loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, pred_tensor, target_tensor):
    return self.neg_loss(pred_tensor, target_tensor)
def _gather_feat_(feat, ind, mask=None):
    dim  = feat.size(2)
    #pdb.set_trace()
    ind  = ind.expand(ind.size(0), ind.size(1), dim)
    #pdb.set_trace()
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    # feat.shape =[2,256,256,2]
    #pdb.set_trace() # ind = torch.Size([2, 128, 2]), feat = [2,256*256,2]
    ind  = ind.unsqueeze(3).expand(ind.size(0), ind.size(1),ind.size(2), dim) # ind = [2,128,2,2]
    #ind = ind.permute(0, 1, 3, 2).contiguous() # ind = [2,128,6,256]
    feat_s = feat.gather(1, ind[:, :, 0, :]) # feat = [2,256*256,2],# ind = [2,128,2]
    feat_e = feat.gather(1, ind[:, :, 1, :])
    feat = (feat_s + feat_e) / 2 # optimize in process
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):

    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, pred, mask, ind, target):
    #pdb.set_trace()
    pred = _transpose_and_gather_feat(pred, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    #target = target.unsqueeze(2).expand_as(mask).float()
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y
def _relu(x):
    y = torch.clamp(x.relu_(), min = 0., max=179.99)
    return y

class CtdetLoss(torch.nn.Module):
    def __init__(self, loss_weight):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.crit_clid = RegL1Loss()
        self.crit_conid = RegL1Loss()
        self.loss_weight = loss_weight
        
    def forward(self, pred_tensor, target_tensor): 
        hm_weight = self.loss_weight['hm_weight']
        wh_weight = self.loss_weight['wh_weight']
        reg_weight = self.loss_weight['reg_weight']
        hm_loss, wh_loss, off_loss = 0, 0, 0

        pred_tensor['hm'] = _sigmoid(pred_tensor['hm'])
        hm_loss += self.crit(pred_tensor['hm'], target_tensor['hm'])
        if wh_weight > 0:
            wh_loss += self.crit_wh(pred_tensor['h_ud'], target_tensor['reg_mask'],target_tensor['ind'], target_tensor['h_ud'])
         #pdb.set_trace()
        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['reg_y'], target_tensor['reg_mask'],target_tensor['ind'], target_tensor['reg_y'])

        # if clid_weight > 0:
        #     clid_loss += self.crit_clid(pred_tensor['cl_id'], target_tensor['reg_mask'], target_tensor['ind'], target_tensor['cl_id'])
        # # if conid_weight > 0:
        #     conid_loss += self.crit_conid(pred_tensor['con_id'], target_tensor['reg_mask'], target_tensor['ind'], target_tensor['con_id'])

        return hm_weight * hm_loss + wh_weight * wh_loss + reg_weight * off_loss #+ lr_weight * lr_loss


