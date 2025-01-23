import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from .mtp import MTP

# inspired by: [https://geohot.github.io/blog/jekyll/update/2021/10/29/an-architecture-for-life.html]
# models/path_planner_gru.pt
class PathPlannerRNN(nn.Module):
  def __init__(self, hidden_size, n_layers=2, n_paths=5, use_mdn=False):
    super(PathPlannerRNN, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    
    # TODO: freeze during RL (+ explore = 0)
    # representation function
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))

    # temporal model
    # self.dynamics = nn.LSTM(1411, hidden_size, n_layers)
    self.dynamics = nn.GRU(1411, hidden_size, n_layers, batch_first=True) # GRU for speed

    # policy model
    # multimodal (multiple paths with probabilities) output (check out mixture density networks)
    # meaning output is M future paths (for now) <xi,yi> for i in range(2*H)
    # along with each path's probabilities, these probabilities are passed through a softmax layer
    self.policy = MTP(hidden_size, n_modes=self.n_paths, use_mdn=use_mdn)
    # TODO: add MDN??

  def forward(self, x_3d, desire):
    feats = []
    for t in range(x_3d.size(1)):
      x = self.vision(x_3d[:, t, :, :, :])
      x = x.view(-1, self.num_flat_features(x))
      x = torch.cat((x, desire), 1)
      feats.append(x.unsqueeze(1))
    x = torch.cat(feats, dim=1)

    lstm_out, _ = self.dynamics(x)
    out = self.policy(lstm_out[:, -1, :])
    return out
  
  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features
