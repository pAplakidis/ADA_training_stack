#!/usr/bin/env python3
import cv2
import time
import random
import io
from collections import deque
from datetime import datetime
import plotly.io as pio
import plotly.graph_objects as go

import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from env import CarlaEnv
from loss import *
from mtp_utils import _get_trajectory_and_modes, calc_steering_angle

def _figshow(fig):
  buf = io.BytesIO()
  pio.write_image(fig, buf)
  buf.seek(0)
  file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  cv2.imshow("Predicted Path", img)


class DrivingAgent:
  def __init__(self, id, show_display=False):
    # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.device = torch.device("cpu") # TODO: run CARLA server on different device to use GPU here
    print("[+] Using device:", self.device)

    self.id = id
    self.show_display = show_display
    self.loss_func = MTPLoss(N_MODES)

    # path planner + controller parameters
    self.desire = np.array([1.0, 0.0, 0.0])  # only forward for now
    self.K = 0.1

    self.model = PathPlannerRNN(HIDDEN_SIZE, N_GRU_LAYERS)
    self.model = load_model(MODEL_PATH, self.model, cpu=True)
    self.model.to(self.device)

    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    self.writer = SummaryWriter(f"runs/{MODEL_NAME}-{datetime.now()}-{int(time.time())}")

    self.terminate = False
    self.last_logged_episode = 0
    self.training_init = False

  def get_action(self, frames):
    steering_angle = 0.0

    with torch.no_grad():
      DES = torch.as_tensor(self.desire).unsqueeze(0).float().to(self.device)
      out = self.model(frames, DES)

      trajectories, modes = _get_trajectory_and_modes(out)
      trajectories = trajectories.cpu().numpy()
      modes = modes.cpu().numpy()
      xy_path = trajectories[0][0]
      for idx, pred_path in enumerate(trajectories[0]):
        if modes[0][idx] == np.max(modes[0]):
          xy_path = trajectories[0][idx]

      steering_angle = self.K * calc_steering_angle(xy_path[0], xy_path[10])
      print(f"[agent]: steering_angle={steering_angle}")

    return steering_angle, xy_path

  def update_replay_memory(self, transition):
    # transition = (current_state, action, reward, new_state, done)
    self.replay_memory.append(transition)

  def train(self):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    # shape = (batch_size, n_frames, C, H, W)
    # FIXME: transition[0] already a tensor, concatenate instead
    current_states = torch.as_tensor(np.array([transition[0] for transition in minibatch])).float().to(self.device)

    x = []
    y = []


def run(id, carla_instance, show_preview=SHOW_DISPLAY, model_path=MODEL_PATH, train=True):
  fig = go.FigureWidget()
  fig.update_layout(xaxis_range=[-50,50])
  fig.update_layout(yaxis_range=[0,50])

  agent = DrivingAgent(id)
  xy_path = np.zeros((TRAJECTORY_LENGTH, N_COORDINATES))

  while True:
    try:
      env = CarlaEnv(carla_instance)
      break
    except Exception as e:
      print(e)
      print("[!] Make sure you are running start_carla.sh")
      print("[*] Restarting ...\n")
      time.sleep(1)

  try:
    for episode in range(EPISODES):
      print(f"\n[+] Episode {episode+1}/{EPISODES}")
      env.reset()
      next_action = 0.0
      prev_observation = None

      idx = 0
      while True:
        # handle step
        observation_frames, reward, done, _ = env.step(next_action)

        if len(observation_frames) < N_FRAMES:
          continue
        if prev_observation is None:
          prev_observation = observation_frames.copy()
          continue

        transition = [prev_observation, next_action, observation_frames, reward, done]
        agent.update_replay_memory(transition)

        # display
        if show_preview:
          cv2.imshow("Display IMG", env.display_img)
          cv2.waitKey(1)

          if fig:
            fig.data = []
            path_x = xy_path[:, 0]
            path_y = xy_path[:, 1]
            marker = {"color": "blue"}  # TODO: color from green to red based on path probability
            fig.add_scatter(x=path_x, y=path_y, name="path", marker=marker)

            if xy_path is not None:
              _figshow(fig)

        # prepare observation
        frames = observation_frames.copy()
        frames = torch.as_tensor(np.array(frames)).unsqueeze(0).float().to(agent.device)
        print(f"\nepisode {episode+1} - step {idx+1} - reward = {reward} - observation = {frames.shape}")

        if done:
          break

        # prepare next step
        # agent.update_replay_memory()
        next_action, xy_path = agent.get_action(frames)
        prev_observation = observation_frames.copy()
        idx += 1
        time.sleep(1)

  except RuntimeError as re:
    print("[!]", re)
    print("Restarting ...")
  finally:
    env.destroy_agents()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  run(0, None)
