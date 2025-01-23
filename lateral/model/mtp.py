import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# MTP code from [https://github.com/daeheepark/PathPredictNusc]
class MTP(nn.Module):
  # n_modes: number of paths output
  # path_len: number of points of each path
  def __init__(self, in_feats, n_modes=3, path_len=200, hidden_feats=4096, use_mdn=True):
    super(MTP, self).__init__()
    self.use_mdn = use_mdn

    self.n_modes = n_modes
    self.fc1 = nn.Linear(in_feats, hidden_feats)
    if self.use_mdn:
      self.bn1 = nn.BatchNorm1d(hidden_feats)
      self.relu = nn.ReLU()

    if self.use_mdn:
      self.fc2 = nn.Linear(hidden_feats, hidden_feats)
      self.mdn = MDN(hidden_feats, int(n_modes * (path_len*2) + n_modes))
    else:
      self.fc2 = nn.Linear(hidden_feats, int(n_modes * (path_len*2) + n_modes))

  def forward(self, x):
    if self.use_mdn:
      pi, sigma, mu = self.mdn(self.fc2(self.relu(self.bn1(self.fc1(x)))))
      mode_probs = mu[:, -self.n_modes:].clone()

      sigma = sigma[:, :-self.n_modes]
      mu = mu[:, :-self.n_modes]
      x = torch.cat((pi, sigma, mu), 1)
    else:
      x = self.fc2(self.fc1(x))
      mode_probs = x[:, -self.n_modes:].clone()
      x = x[:, :-self.n_modes]

    # normalize the probabilities to sum to 1 for inference
    if not self.training:
      mode_probs = F.softmax(mode_probs, dim=1)

    return torch.cat((x, mode_probs), 1)
