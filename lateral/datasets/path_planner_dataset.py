#!/usr/bin/env python3
import cv2
import numpy as np

from torch.utils.data import Dataset

from util import *
from model import *
from dataset_utils import *

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
