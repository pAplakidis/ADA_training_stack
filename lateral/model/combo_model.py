import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from .mtp import MTP


class ComboModel(nn.Module):
  def __init__(self, n_paths=5, use_mdn=True):
    super(ComboModel, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    self.policy = MTP(1411, n_modes=self.n_paths, use_mdn=use_mdn)
    self.cr_detector = nn.Sequential(
      nn.Linear(1411, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Linear(128, 84),
      nn.BatchNorm1d(84),
      nn.ReLU(),
      nn.Linear(84, 1)
    )

  def forward(self, x, desire):
    x = self.vision(x)
    x = x.view(-1, self.num_flat_features(x))
    x = torch.cat((x, desire), 1)
    #print(x.shape)
    path = self.policy(x)
    crossroad = torch.sigmoid(self.cr_detector(x))
    return path, crossroad

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features
