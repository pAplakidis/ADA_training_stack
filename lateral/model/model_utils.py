import os
import torch

def save_model(path, model):
  directory = os.path.join(*path.split('/')[:-1])
  if not os.path.exists(directory): os.makedirs(directory)
  torch.save(model.state_dict(), path)
  print("[+] Model saved at", path)

def load_model(path, model):
  model.load_state_dict(torch.load(path))
  print("[+] Loaded model from", path)
  return model