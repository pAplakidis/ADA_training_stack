import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2

# MDN code from [https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py]
class MDN(nn.Module):
  def __init__(self, in_feats, out_feats, n_gaussians=1):
    super(MDN, self).__init__()
    self.in_feats = in_feats
    self.out_feats = out_feats
    self.n_gaussians = n_gaussians
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


class ComboModel(nn.Module):
  def __init__(self, n_paths=5, use_mdn=True):
    super(ComboModel, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(pretrained=True)

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


# Input: 2 consecutive frames
# Output: trajectory and crossroad prediction
class SuperComboModel(nn.Module):
  def __init__(self, input_size=6, hidden_size=128, n_layers=1, n_paths=5):
    super(SuperComboModel, self).__init__()
    self.n_paths = n_paths
    self.input_size = input_size    # input channels (2 bgr frames -> 2*3 channels)
    self.hidden_size = hidden_size  # output size of GRU unit
    self.n_layers = n_layers        # number of layers in GRU unit
    effnet = efficientnet_b2(pretrained=True)
    effnet.features[0][0] = nn.Conv2d(self.input_size, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    # like MuZero: vision->representation(h), state->dynamics(g), policy->prediction(f)
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    # self.vision = efficientnet_b2(pretrained=True)
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

# TODO: try GRU for speed
# inspired by: [https://geohot.github.io/blog/jekyll/update/2021/10/29/an-architecture-for-life.html]
class PathPlannerLSTM(nn.Module):
  def __init__(self, hidden_size, n_layers=2, n_paths=5, use_mdn=False):
    super(PathPlannerLSTM, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(pretrained=True)
    
    # representation model
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))

    # dynamics model
    self.dynamics = nn.LSTM(1411, hidden_size, n_layers)

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
