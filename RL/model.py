import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


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
    effnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
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
    effnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    # freeze during RL (+ explore = 0, i.e. always on-policy)
    for param  in effnet.parameters():
      param.requires_grad = False
    
    # representation function
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))

    # temporal model
    # self.dynamics = nn.LSTM(1411, hidden_size, n_layers)
    self.dynamics = nn.GRU(1411, hidden_size, n_layers, batch_first=True) # GRU for speed

    # policy model
    # multimodal (multiple paths with probabilities) output (check out mixture density networks)
    # meaning output is M future paths (for now) <xi,yi> for i in range(2*H)
    # along with each path's probabilities, these probabilities are passed through a softmax layer
    self.policy = MTP(hidden_size, n_modes=self.n_paths)

  def forward(self, x_3d, desire):
    feats = []
    for t in range(x_3d.size(1)):
      x = self.vision(x_3d[:, t, :, :, :])
      x = x.view(-1, self.num_flat_features(x))
      x = torch.cat((x, desire), 1)
      feats.append(x.unsqueeze(1))
    x = torch.cat(feats, dim=1)

    temporal_out, _ = self.dynamics(x)
    out = self.policy(temporal_out[:, -1, :])
    return out
  
  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


# input: sequence of frames, car desire (state), actor model output - probs + trajectories (action)
# output: Q-value of (state, action)
class CriticModel(nn.Module):
  def __init__(self, vision_backbone, hidden_size, n_paths=2, n_layers=2):
    super(CriticModel, self).__init__()

    self.vision = vision_backbone
    self.dynamics = nn.GRU(1411, hidden_size, n_layers, batch_first=True) # GRU for speed
    self.value_head = nn.Linear(hidden_size + 2005, 1)

  def forward(self, x_3d, desire, model_action):
    # handle feats like driving model
    feats = []
    for t in range(x_3d.size(1)):
      x = self.vision(x_3d[:, t, :, :, :])
      x = x.view(-1, self.num_flat_features(x))
      x = torch.cat((x, desire), 1)
      feats.append(x.unsqueeze(1))
    x = torch.cat(feats, dim=1)

    temporal_out, _ = self.dynamics(x)

    # calculate Q-value
    x = torch.cat((temporal_out[:, -1, :], model_action), 1)
    value = self.value_head(x)
    return value

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

def load_model(path, model, cpu=False):
  if cpu:
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  else:
    model.load_state_dict(torch.load(path))
  print("Loaded model from", path)
  return model

