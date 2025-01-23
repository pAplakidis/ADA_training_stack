import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

def gaussian_probability(sigma, mu, target):
  target = target.unsqueeze(1).expand_as(sigma)
  ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
  return torch.prod(ret, 2)

# not used
def MDNLoss(pi, sigma, mu, target):
  prob = pi * gaussian_probability(sigma, mu, target)
  nll = -torch.log(torch.sum(prob, dim=1))
  return torch.mean(nll)

# TODO: study this loss better from the paper and double-check it
def mdn_loss(pi, sigma, mu, target):
  exponent = -0.5 * ((target - mu) / (sigma + 1e-6)) ** 2
  normalizer = 1.0 / (torch.sqrt(2 * torch.tensor(torch.pi, dtype=torch.float32)) * sigma)
  pdf = normalizer * torch.exp(exponent)
  weighted_sum = torch.sum(pi * pdf, dim=1)
  loss = -torch.log(weighted_sum + 1e-6)
  return torch.mean(loss)
