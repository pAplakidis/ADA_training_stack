import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# MDN code from [https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py]
class MDN(nn.Module):
  def __init__(self, in_feats, out_feats, n_gaussians=1):
    super(MDN, self).__init__()
    self.in_feats = in_feats
    self.out_feats = out_feats
    self.n_gaussians = n_gaussians
    # TODO: make constants in utils
    self.n_modes = 5
    self.n_points = 200
    self.pi = nn.Sequential(
      nn.Linear(self.in_feats, self.n_modes * self.n_points * self.n_gaussians),
      nn.Softmax(dim=1)
    )
    self.sigma = nn.Linear(self.in_feats, self.out_feats * self.n_gaussians)
    self.mu = nn.Linear(self.in_feats, self.out_feats * self.n_gaussians)

  def forward(self, x):
    pi = self.pi(x)
    sigma = torch.exp(self.sigma(x))
    sigma = sigma.view(-1, self.n_gaussians, self.out_feats)
    mu = self.mu(x)
    mu = mu.view(-1, self.n_gaussians, self.out_feats)
    return pi, sigma[:, 0], mu[:, 0]
