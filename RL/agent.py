#!/usr/bin/env python3
import cv2
import time
import random
import io
import plotly.io as pio
import plotly.graph_objects as go
from tqdm import trange 
from collections import deque
from datetime import datetime

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
    self.id = id
    self.show_display = show_display

    self.device = torch.device("cpu") # NOTE: CARLA takes GPU, making CUDA unavailable for training
    print("[+] Using device:", self.device)

    # path planner + controller parameters
    self.desire = np.array([1.0, 0.0, 0.0])  # only forward for now
    self.K = 0.1

    # load pretrained actor model
    self.actor_model = PathPlannerRNN(HIDDEN_SIZE, N_GRU_LAYERS)
    self.actor_model = load_model(MODEL_PATH, self.actor_model, cpu=True)
    self.actor_model.to(self.device)
    # TODO: train alongside critic model
    # init critic model
    # self.critic_model = CriticModel()
    # self.critic_model.to(self.device)

    self.loss_func = MTPLoss(N_MODES)
    self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=LR) # TODO: check out RMSProp

    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    self.writer = SummaryWriter(f"runs/{MODEL_NAME}-{datetime.now()}-{int(time.time())}-BS_{MINIBATCH_SIZE}-LR_{LR}-HS_{HIDDEN_SIZE}-N_GRU_{N_GRU_LAYERS}")

  def get_action(self, frames):
    steering_angle = 0.0

    with torch.no_grad():
      self.actor_model.eval()
      DES = torch.as_tensor(self.desire).unsqueeze(0).float().to(self.device)
      out = self.actor_model(frames, DES)

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

  def train(self, steps_cnt):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
    # if len(self.replay_memory) < 2 * MINIBATCH_SIZE or len(self.replay_memory) % MINIBATCH_SIZE == 0:
      return
    
    print("[*] Training agent")
    self.actor_model.train()
    actor_losses = []
    for i in (t := trange(0, len(self.replay_memory), MINIBATCH_SIZE)):
      minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

      # transition = (current_state, action, reward, new_state, done)
      # state_shape = (batch_size, n_frames, C, H, W)
      current_states = torch.as_tensor(np.array([transition[0] for transition in minibatch])).float().to(self.device)
      rewards = torch.as_tensor(np.array([transition[2] for transition in minibatch])).float().to(self.device)
      print(rewards.shape)
      print(current_states.shape)
      discounted_reward = self.calc_reward(rewards, DISCOUNT)

      self.actor_optim.zero_grad()
      DES = torch.as_tensor(self.desire).repeat(MINIBATCH_SIZE, 1).float().to(self.device)
      out = self.actor_model(current_states, DES)

      trajectories, modes = _get_trajectory_and_modes(out)
      # xy_path = trajectories[0][0]
      # for idx, pred_path in enumerate(trajectories[0]):
      #   if modes[0][idx] == np.max(modes.detach().cpu().numpy()[0]):
      #     xy_path = trajectories[0][idx]

      print(modes.shape, discounted_reward.shape)
      actor_loss = -torch.log(torch.max(modes, dim=1).values) * discounted_reward  # FIXME: this is the loss for path classification, we also need trajectory loss
      actor_loss = actor_loss.mean()
      actor_loss.backward()
      self.actor_optim.step()

      actor_losses.append(actor_loss.item())
      t.set_description("R(τ): %.2f - actor_loss: %.2f"%(discounted_reward.mean().item(), actor_loss.item()))

    self.writer.add_scalar("episode actor loss", np.array(actor_losses).mean(), steps_cnt)
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

  @staticmethod
  def calc_reward(rewards, gamma):
    r = 0
    final_rewards = np.zeros_like(rewards)
    for i, reward in enumerate(reversed(rewards)):
      r +=  (gamma ** i) * reward
      final_rewards[i] = r
    return torch.as_tensor(final_rewards)


# TODO: multiple maps
def run(id, carla_instance, show_preview=SHOW_DISPLAY, model_path=MODEL_PATH, train=True):
  fig = go.FigureWidget()
  fig.update_layout(xaxis_range=[-50,50])
  fig.update_layout(yaxis_range=[0,50])

  agent = DrivingAgent(id)
  xy_path = np.zeros((TRAJECTORY_LENGTH, N_COORDINATES))

  all_rewards = []
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
    total_step_cnt = 0
    for episode in range(EPISODES):
      print(f"\n[=>] Episode {episode+1}/{EPISODES}")
      env.reset()
      next_action = 0.0
      prev_observation = None

      episode_rewards = []
      idx = 0
      while True:
        # handle step
        observation_frames, reward, done, _ = env.step(next_action)
        episode_rewards.append(reward)

        if len(observation_frames) < N_FRAMES:
          continue
        if prev_observation is None:
          prev_observation = observation_frames.copy()
          continue

        transition = [prev_observation, next_action, reward, observation_frames, done]
        agent.update_replay_memory(transition)

        # display
        if show_preview:
          cv2.imshow("Display IMG", env.display_img)
          cv2.waitKey(1)

          if fig:
            fig.data = []
            path_x = xy_path[:, 0]
            path_y = xy_path[:, 1]
            marker = {"color": "blue"}  # TODO: color gradient from green to red based on path probability
            fig.add_scatter(x=path_x, y=path_y, name="path", marker=marker)

            if xy_path is not None:
              _figshow(fig)

        # prepare observation
        frames = observation_frames.copy()
        frames = torch.as_tensor(np.array(frames)).unsqueeze(0).float().to(agent.device)
        print(f"\nepisode {episode+1} - step {idx+1} - reward = {reward} - observation = {frames.shape}")
        agent.writer.add_scalar("running reward", reward, total_step_cnt)

        if done:
          print("[*] Episode done")
          env.destroy_agents()
          break

        # prepare next step
        # agent.update_replay_memory()
        next_action, xy_path = agent.get_action(frames)
        prev_observation = observation_frames.copy()

        idx += 1
        total_step_cnt += 1
        time.sleep(1)

      agent.writer.add_scalar("episode reward", np.array(episode_rewards).sum(), episode)
      all_rewards.append(episode_rewards)
      agent.train(total_step_cnt)
  except KeyboardInterrupt:
    print("[~] Training stopped by user")

  env.destroy_agents()
  cv2.destroyAllWindows()
  save_model(MODEL_SAVE_PATH, agent.actor_model)


if __name__ == "__main__":
  run(0, None)
