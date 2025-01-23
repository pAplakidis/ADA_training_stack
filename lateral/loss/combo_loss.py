import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtp_loss import MTPLoss


class ComboLoss(nn.Module):
  def __init__(self, task_num, model, device, use_mdn=True):
    super(ComboLoss, self).__init__()
    self.task_num = task_num  # TODO: maybe make this constant
    self.model = model
    self.device = device
    self.log_vars = nn.Parameter(torch.zeros((task_num)))

    self.path_loss = MTPLoss(self.model.n_paths, use_mdn=use_mdn)
    self.cr_loss = nn.BCELoss()

  def forward(self, preds, ground_truths):
    path_pred, cr_pred = preds
    path_gt, cr_gt = ground_truths

    loss0 = self.path_loss(path_pred, path_gt)
    loss1 = self.cr_loss(cr_pred, cr_gt)

    # TODO: need better multitask loss (weighted sum maybe)
    precision0 = torch.exp(-self.log_vars[0])
    #loss0 = precision0*loss0 + self.log_vars[0]

    precision1 = torch.exp(-self.log_vars[1])
    #loss1 = precision1*loss1 + self.log_vars[1]

    loss = loss0 + loss1
    #loss = loss.mean()

    return loss.to(self.device), loss0, loss1

