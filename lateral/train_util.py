#!/usr/bin/env python3
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from datetime import date

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler

from util import *
from model import *
from renderer import *

# display resolution
d_W = 1920 // 2
d_H = 1080 // 2


class PathPlannerDataset(Dataset):
  def __init__(self, base_dir):
    super(Dataset, self).__init__()
    self.base_dir = base_dir
    self.video_path = base_dir + "video.mp4"
    self.poses_path = base_dir + "poses.npy"
    self.framepath_path = base_dir + "frame_paths.npy"

    # load meta-data (poses, paths, etc)
    self.poses = np.load(self.poses_path)
    self.frame_paths = np.load(self.framepath_path)
    self.local_poses, self.local_path, self.local_orientations = get_relative_poses(self.poses)
    #print(self.local_path.shape)
    print("Frame Paths (2D):", self.frame_paths.shape)

    # load video
    self.cap = cv2.VideoCapture(self.video_path)

  def __len__(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - LOOKAHEAD

  def __getitem__(self, idx):
    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)
    self.cap.set(1, idx)
    ret, frame = self.cap.read()
    frame = cv2.resize(frame, (W,H))
    frame = np.moveaxis(frame, -1, 0)
    # TODO: use path for now, later on predict poses
    # TODO: this is a tempfix, we need to cleanup data (either check for nan during data-collection or training)
    if np.isnan(self.frame_paths[idx]).any():
      self.frame_paths[idx] = np.zeros_like(self.frame_paths[idx])
    return {"image": frame, "path": self.frame_paths[idx]}


# BUG: WHEN USING MDN: getting nan from loss, meaning some tensors return nan here
#      (not due to race condition, this code is the problem)... FIX
# TODO: modify this (switch between single/multi-frame for RNN)
def custom_collate_pathplanner(batch):
  collated_batch = {
    "image": [],
    "path": torch.stack([torch.tensor(sample["path"]) for sample in batch]),
    "desire": torch.stack([torch.tensor(sample["desire"]) for sample in batch]),
  }

  for item in batch:
    frame = load_frame(item)
    collated_batch["image"].append(frame)
  collated_batch["image"] = torch.stack([torch.tensor(frame) for frame in collated_batch["image"]])

  return collated_batch

def custom_collate_combomodel(batch):
  collated_batch = {
    "image": [],
    "path": torch.stack([torch.tensor(sample["path"]) for sample in batch]),
    "desire": torch.stack([torch.tensor(sample["desire"]) for sample in batch]),
    "crossroad": torch.stack([torch.tensor(sample["crossroad"]) for sample in batch])
  }

  for item in batch:
    frame = load_frame(item)
    collated_batch["image"].append(frame)
  collated_batch["image"] = torch.stack([torch.tensor(frame) for frame in collated_batch["image"]])

  return collated_batch

def load_frame(item):
  cap = cv2.VideoCapture(item["video_path"])
  cap.set(cv2.CAP_PROP_POS_FRAMES, item["framenum"])
  _, frame = cap.read()
  frame = cv2.resize(frame, (W, H))
  frame = np.moveaxis(frame, -1, 0)
  return frame

def load_frames(item):
  cap = cv2.VideoCapture(item["video_path"])
  cap.set(cv2.CAP_PROP_POS_FRAMES, item["framenum"])
  frames = []
  for _ in range(2):
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.resize(frame, (W, H))
    frame = np.moveaxis(frame, -1, 0)
    frames.append(frame)
  return frames


class MultiVideoDataset(Dataset):
  def __init__(self, base_dir, multi_frames=False, combo=True):
    super(Dataset, self).__init__()
    self.mutli_frames = multi_frames
    self.combo = combo

    # directories
    self.base_dir = base_dir
    self.video_paths = []
    self.framepath_paths = []
    self.desires_paths = []
    self.crossroads_paths = []
    self.input_frames = [np.zeros((3, W, H)) for _ in range(2)] # 2 consecutive frames for GRU

    print("Data from:")
    for dir in sorted(os.listdir(base_dir)):
      prefix = self.base_dir+dir+"/"
      print(prefix)
      self.video_paths.append(prefix+"video.mp4")
      self.framepath_paths.append(prefix+"frame_paths.npy")
      self.desires_paths.append(prefix+"desires.npy")
      if self.combo:
        self.crossroads_paths.append(prefix+"crossroads.npy")

    # load and index actual data
    self.caps = [cv2.VideoCapture(str(video_path)) for video_path in self.video_paths]
    self.images = [[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-LOOKAHEAD+1)]
    self.frame_paths = [np.load(fp) for fp in self.framepath_paths]
    self.desires = [np.load(desires) for desires in self.desires_paths]
    for i in range(len(self.desires)):
      self.desires[i] = one_hot_encode(self.desires[i])
    if self.combo:
      self.crossroads = [np.load(crds) for crds in self.crossroads_paths]
    """
    # check length of images and paths
    print("images:")
    cap = 0
    frame_cnt = []
    for i in range(len(self.images)):
      if self.images[i][0] != cap:
        cap += 1
        frame_cnt.append(self.images[i-1][1])
    frame_cnt.append(self.images[-1][1])
    print(frame_cnt)
    print("frame paths:")
    for fp in self.frame_paths:
      print(fp.shape)
    """

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    """
    if self.mutli_frames:
      if idx != 0:
        # get previous frame
        capid, framenum = self.images[idx-1]
        cap = self.caps[capid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        _, frame1 = cap.read()

        frame1 = cv2.resize(frame1, (W,H))
        frame1 = np.moveaxis(frame1, -1, 0)
        self.input_frames[0] = frame1

      # get current frame
      capid, framenum = self.images[idx]
      cap = self.caps[capid]
      cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
      ret, frame2 = cap.read()

      frame2 = cv2.resize(frame2, (W,H))
      frame2 = np.moveaxis(frame2, -1, 0)
      self.input_frames[1] = frame2
    else:
      # get frame
      capid, framenum = self.images[idx]
      cap = self.caps[capid]
      cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
      _, frame = cap.read()

      frame = cv2.resize(frame, (W,H))
      frame = np.moveaxis(frame, -1, 0)
    """

    capid, framenum = self.images[idx]
    path = self.frame_paths[capid][framenum]
    if np.isnan(path).any():
      path = np.zeros_like(path)
    desire = self.desires[capid][framenum]
    if self.combo:
      crossroad = self.crossroads[capid][framenum]

    """
    if self.mutli_frames:
      if self.combo:
        payload = {
          "images": self.input_frames,
          "path": path,
          "desire": desire,
          "crossroad": crossroad
        }
      else:
        payload = {
          "images": self.input_frames,
          "path": path,
          "desire": desire,
        }
    else:
      if self.combo:
        payload = {
          "image": frame,
          "path": path,
          "desire": desire,
          "crossroad": crossroad
        }
      else:
        payload = {
          "image": frame,
          "path": path,
          "desire": desire,
        }
    return payload
    """

    item = {
      "video_path": self.video_paths[capid],
      "framenum": framenum,
      "path": path,
      "desire": desire
    }
    if self.combo:
      item["crossroad"] = crossroad
    
    return item


class Trainer:
  def __init__(self, device, model, train_loader, val_loader, model_path, writer_path=None, eval_epoch=False, use_rnn=False, combo=True):
    self.use_rnn = use_rnn  # switch training RNN or CNN
    self.combo = combo      # switch PathPlanner or ComboModel/multitask
    self.eval_epoch = eval_epoch
    self.model_path = model_path
    if not writer_path:
      today = str(date.today())
      writer_path = "runs/" + today
    print("[+] Tensorboard output path:", writer_path)

    self.writer = SummaryWriter(writer_path)
    self.device = device
    print("[+] Device:", self.device)
    self.model = model.to(self.device)
    self.train_loader = train_loader
    self.val_loader = val_loader

  def save_checkpoint(state, path):
    torch.save(state, path)
    print("Checkpoint saved at", path)

  def train(self, epochs=100, lr=1e-4, use_mdn=True):
    NANS = 0
    #loss_func = nn.MSELoss()
    if self.combo:
      loss_func = ComboLoss(2, self.model, self.device, use_mdn=use_mdn)
    else:
      loss_func = MTPLoss(self.model.n_paths, use_mdn=use_mdn)
    optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
    # scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.99)

    # evaluate model
    def eval(val_losses, train=False):
      if not train:
        print("[+] Evaluating ...")
      with torch.no_grad():
        try:
          self.model.eval()
          for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
            if self.use_rnn:
              IN_FRAMES = sample_batched["images"]
              for i in range(2):
                IN_FRAMES[i] = torch.as_tensor(IN_FRAMES[i]).float().to(self.device)
            else:
              X = torch.as_tensor(sample_batched["image"]).float().to(self.device)
            desire = torch.as_tensor(sample_batched["desire"]).float().to(self.device)
            Y_path = torch.as_tensor(sample_batched["path"]).float().to(self.device)
            if self.combo:
              Y_cr = torch.as_tensor(sample_batched["crossroad"]).float().to(self.device)

            if self.combo:
              if self.use_rnn:
                out_path, out_cr = self.model(IN_FRAMES, desire)
              else:
                out_path, out_cr = self.model(X, desire)
              loss = loss_func([out_path, out_cr], [Y_path, Y_cr])
            else:
              out_path = self.model(X, desire)
              loss = loss_func(out_path, Y_path)

            if not torch.isnan(loss):
              if not train:
                self.writer.add_scalar('running evaluation loss', loss.item(), i_batch)
              val_losses.append(loss.item())
            t.set_description("Eval Batch Loss: %.2f"%(loss.item()))

        except KeyboardInterrupt:
          print("[~] Evaluation stopped by user")
      if not train:
        print("[+] Evaluation Done")
      return val_losses

    # TODO: add checkpoints so that we can resume training if interrupted
    # train model
    losses = []
    vlosses = []
    try:
      print("[+] Training ...")
      idx = 0
      for epoch in range(epochs):
        self.model.train()
        print("\n[=>] Epoch %d/%d"%(epoch+1, epochs))
        epoch_losses = []
        epoch_vlosses = []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          if self.use_rnn:
            IN_FRAMES = sample_batched["images"]
            for i in range(2):
              IN_FRAMES[i] = torch.as_tensor(IN_FRAMES[i]).float().to(self.device)
          else:
            X = torch.as_tensor(sample_batched["image"]).float().to(self.device)
          desire = torch.as_tensor(sample_batched["desire"]).float().to(self.device)
          Y_path = torch.as_tensor(sample_batched["path"]).float().to(self.device)
          if self.combo:
            Y_cr = torch.as_tensor(sample_batched["crossroad"]).float().to(self.device)

          optim.zero_grad(set_to_none=True)
          if self.combo:
            if self.use_rnn:
              out_path, out_cr = self.model(IN_FRAMES, desire)
            else:
              out_path, out_cr = self.model(X, desire)
            loss = loss_func([out_path, out_cr], [Y_path, Y_cr])
          else:
            out_path = self.model(X, desire)
            loss = loss_func(out_path, Y_path)

          if not torch.isnan(loss):
            self.writer.add_scalar("running loss", loss.item(), idx)
            epoch_losses.append(loss.item())
            loss.backward()
            optim.step()
          else:
            NANS += 1
            """
            print("NaN Loss Detected!")
            print(X)
            print(desire)
            print(Y_path)
            exit(0)
            """

          t.set_description("Batch Training Loss: %.2f"%(loss.item()))
          idx += 1

        avg_epoch_loss = np.array(epoch_losses).mean()
        losses.append(avg_epoch_loss)
        self.writer.add_scalar("epoch training loss", avg_epoch_loss, epoch)
        print("[->] Epoch average training loss: %.4f"%(avg_epoch_loss))
        # scheduler.step()

        if self.eval_epoch:
          epoch_vlosses = eval(epoch_vlosses, train=True)
          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          vlosses.append(avg_epoch_vloss)
          # TODO: custom plot on the same figure as final training losses
          self.writer.add_scalar('epoch evaluation loss', avg_epoch_vloss, epoch)
          print("[->] Epoch average evaluation loss: %.4f"%(avg_epoch_vloss))

    except KeyboardInterrupt:
      print("[~] Training stopped by user")
    print("[+] Training Done")
    save_model(self.model_path, self.model)
    print("NaN losses detected:", NANS)

    # final evaluation
    val_losses = []
    val_losses = eval(val_losses)
    print("Avg Eval Loss: %.4f"%(np.array(val_losses).mean()))
  
    self.writer.close()
    return self.model


if __name__ == "__main__":
  #renderer = Renderer3D(RW, RH)
  """
  # Test single video dataset
  dataset = PathPlannerDataset("../data/sim/22/")
  print(len(dataset))
  samp = dataset[100]
  img, path = samp["image"], samp["path"]
  print(img.shape)
  print(path.shape)
  print(path)

  # plot path
  fig = go.FigureWidget()
  fig.add_scatter()
  fig.update_layout(xaxis_range=[-50,50])
  fig.update_layout(yaxis_range=[0,50])
  fig.data[0].x = path[:, 0]
  fig.data[0].y = path[:, 1]
  figshow(fig)

  disp_img = np.moveaxis(img, 0, -1)
  disp_img = cv2.resize(disp_img, (d_W,d_H))
  print(disp_img.shape)

  #renderer.draw(path)

  #draw_path(path, disp_img)
  cv2.imshow("DISPLAY", disp_img)
  cv2.waitKey(0)

  dataset.cap.release()
  cv2.destroyAllWindows()
  """

  # Test multi-video dataset
  dataset = MultiVideoDataset("../data/sim/train/")
  print("Frames in dataset:", len(dataset))
  idxs = []
  for _ in range(10):
    idxs.append(random.randint(0, len(dataset)))

  for idx in idxs:
    print("[+] Frame:", idx)
    samp = dataset[idx]
    img, path, desire, crossroad = samp["image"], samp["path"], samp["desire"], samp["crossroad"]
    print(img.shape)
    print(path.shape)
    desire_idx = np.argmax(desire)
    print("Desire:", desire_idx, "=>", DESIRE[desire_idx])
    print("Crossroad:", crossroad, "=>", CROSSROAD[crossroad])

    # plot path
    fig = go.FigureWidget()
    fig = go.FigureWidget()
    fig.add_scatter()
    fig.update_layout(xaxis_range=[-50,50])
    fig.update_layout(yaxis_range=[0,50])
    fig.data[0].x = path[:, 0]
    fig.data[0].y = path[:, 1]
    figshow(fig)

    disp_img = np.moveaxis(img, 0, -1)
    disp_img = cv2.resize(disp_img, (d_W, d_H))

    cv2.imshow("DISPLAY", disp_img)
    cv2.waitKey(0)

  # Test batch loader
  """
  loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0)
  for i_batch, sample_batched in enumerate(loader):
    X = torch.tensor(sample_batched['image']).float()
    Y = torch.tensor(sample_batched['path']).float()
    print(i_batch, X.shape, Y.shape)
  """

  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
