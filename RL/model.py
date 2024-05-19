import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2


# MTP code from [https://github.com/daeheepark/PathPredictNusc]
class MTP(nn.Module):
  # n_modes: number of paths output
  # path_len: number of points of each path
  def __init__(self, in_feats, n_modes=3, path_len=200, hidden_feats=4096):
    super(MTP, self).__init__()

    self.n_modes = n_modes
    self.fc1 = nn.Linear(in_feats, hidden_feats)
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


# NOTE: resolutions not divisible by 8, 16 can cause problems
class PathPlanner(nn.Module):
  def __init__(self, n_paths=5, use_mdn=True):
    super(PathPlanner, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(pretrained=True)
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    #del self.vision.classifier
    """
     (classifier): Sequential(                                                                            
      (0): Dropout(p=0.3, inplace=True)                                                                  
      (1): Linear(in_features=1408, out_features=1000, bias=True)
    """
    # multimodal (multiple paths with probabilities) output (check out mixture density networks)
    # meaning output is M future paths (for now) <xi,yi> for i in range(2*H)
    # along with each path's probabilities, these probabilities are passed through a softmax layer
    self.policy = MTP(1411, n_modes=self.n_paths, use_mdn=use_mdn)

  def forward(self, x, desire):
    x = self.vision(x)
    x = x.view(-1, self.num_flat_features(x))
    x = torch.cat((x, desire), 1)
    #print(x.shape)
    x = self.policy(x)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


# inspired by: [https://geohot.github.io/blog/jekyll/update/2021/10/29/an-architecture-for-life.html]
# models/path_planner_gru.pt
class PathPlannerRNN(nn.Module):
  def __init__(self, hidden_size, n_layers=2, n_paths=5, use_mdn=False):
    super(PathPlannerRNN, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(pretrained=True)
    
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


# ===========================

def save_model(path, model):
 torch.save(model.state_dict(), path)
 print("Model saved at", path)

def load_model(path, model):
  model.load_state_dict(torch.load(path))
  print("Loaded model from", path)
  return model

