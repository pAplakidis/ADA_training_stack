import cv2
import torch
import numpy as np

# display resolution
d_W = 1920 // 2
d_H = 1080 // 2

# model input resolution
W = H = 224
N_FRAMES = 5

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

# TODO: spread out the frames (rate too high, frames too similar)
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

