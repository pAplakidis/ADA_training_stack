#!/usr/bin/env python3
import os
import cv2
import random
import numpy as np

import plotly.graph_objects as go

import torch
from torch.utils.data import Dataset

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

# FIXME: images (N_FRAMES)
def custom_collate_pathplanner_lstm(batch):
  collated_batch = {
    "images": [],
    "path": torch.stack([torch.tensor(sample["path"]) for sample in batch]),
    "desire": torch.stack([torch.tensor(sample["desire"]) for sample in batch]),
  }

  for item in batch:
    frames = load_frames(item)
    collated_batch["images"].append(frames)
  collated_batch["images"] = torch.stack([torch.tensor(np.array(frame)) for frame in collated_batch["images"]])

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
  cap.set(cv2.CAP_PROP_POS_FRAMES, item["framenum"]-N_FRAMES)
  frames = []
  for _ in range(N_FRAMES):
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.resize(frame, (W, H))
    frame = np.moveaxis(frame, -1, 0)
    frames.append(frame)
  return frames


class MultiVideoDataset(Dataset):
  def __init__(self, base_dir, multi_frames=False, combo=True, verbose=False):
    super(Dataset, self).__init__()
    self.multi_frames = multi_frames
    self.combo = combo
    self.verbose = verbose

    # directories
    self.base_dir = base_dir
    self.video_paths = []
    self.framepath_paths = []
    self.desires_paths = []
    self.crossroads_paths = []
    self.input_frames = [np.zeros((3, W, H)) for _ in range(2)] # 2 consecutive frames for GRU

    if self.verbose: print("Data from:")
    for dir in sorted(os.listdir(base_dir)):
      prefix = self.base_dir+dir+"/"
      if self.verbose: print(prefix)

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

    # FOR DEBUGGING
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
    # ensure that we don't go out of range (need prev_frames)
    if idx < N_FRAMES:
      idx = N_FRAMES

    capid, framenum = self.images[idx]

    path = self.frame_paths[capid][framenum]
    if np.isnan(path).any():
      path = np.zeros_like(path)
    desire = self.desires[capid][framenum]
    if self.combo:
      crossroad = self.crossroads[capid][framenum]

    # TODO: get previous frames, idx should be > N_IMAGES-1
    if self.multi_frames:
      item = {
        "video_path": self.video_paths[capid],
        "framenum": framenum,
        "path": path,
        "desire": desire,
      }
      if self.combo:
        item["crossroad"] = crossroad
    else:
      item = {
        "video_path": self.video_paths[capid],
        "framenum": framenum,
        "path": path,
        "desire": desire
      }
      if self.combo:
        item["crossroad"] = crossroad
    
    return item


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
