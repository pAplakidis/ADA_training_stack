import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from .mtp import MTP

# Input: 2 consecutive frames
# Output: trajectory and crossroad prediction
class SuperComboModel(nn.Module):
  def __init__(self, input_size=6, hidden_size=128, n_layers=1, n_paths=5):
    super(SuperComboModel, self).__init__()
    self.n_paths = n_paths
    self.input_size = input_size    # input channels (2 bgr frames -> 2*3 channels)
    self.hidden_size = hidden_size  # output size of GRU unit
    self.n_layers = n_layers        # number of layers in GRU unit
    effnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    effnet.features[0][0] = nn.Conv2d(self.input_size, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    # like MuZero: vision->representation(h), state->dynamics(g), policy->prediction(f)
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    # self.vision = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    # self.state = nn.GRU(self.vision._fc.in_features, self.hidden_size, self.n_layers, batch_first=True)
    self.state = nn.GRU(1411, self.hidden_size, self.n_layers, batch_first=True)
    self.policy = MTP(self.hidden_size, n_modes=self.n_paths)
    self.cr_detector = nn.Sequential(
      nn.Linear(hidden_size, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

  def forward(self, in_frames, desire):
    x = torch.cat(in_frames, dim=1)
    x = self.vision(x)
    x = x.view(-1, self.num_flat_features(x))
    x = torch.cat((x, desire), dim=1)
    # print(x.shape)
    out_GRU, pre_GRU = self.state(x)
    # x = out_GRU[:, -1, :] # get the output of the last time step
    path = self.policy(out_GRU)
    crossroad = torch.sigmoid(self.cr_detector(out_GRU))
    return path, crossroad

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

