#!/usr/bin/env python3
import time
import random
from collections import deque
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *


class DrivingAgent:
  def __init__(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[+] Using device:", self.device)

    self.model = PathPlannerRNN(N_GRU_LAYERS).to(self.device)
    # TODO: load model weights as well

    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    self.writer = SummaryWriter(f"runs/{MODEL_NAME}-{datetime.now()}-{int(time.time())}")

    self.terminate = False
    self.last_logged_episode = 0
    self.training_init = False

  # TODO: might not be needed
  def update_replay_memory(self, transition):
    # transition = (current_state, action, reward, new_state, done)
    self.replay_memory.append(transition)

  def train(self):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    # FIXME: 5 frames needed for model
    current_states = torch.as_tensor(np.array([transition[0] for transition in minibatch])).float().to(self.device)

    # TODO: forward observation (5 frames) to model

    x = []
    y = []


if __name__ == "__main__":
  agent = DrivingAgent()
