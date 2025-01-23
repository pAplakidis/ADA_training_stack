#!/usr/bin/env python3
import os
from torch.utils.data import DataLoader, random_split

from config import *
from util import *

from datasets.multi_video_dataset import MultiVideoDataset
from trainer import Trainer

# EXAMPLE USAGE: MODEL_PATH="models/path_planner.pt" WRITER_PATH="runs/test_1" ./train.py

model_path = os.getenv("MODEL_PATH")
if model_path == None:
  model_path = "models/path_planner_desire.pt"
print("[+] Model save path:", model_path)

writer_path = os.getenv("WRITER_PATH")
print("[+] Tensorboard Writer path:", writer_path)


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # define/select model
  if VERBOSE: print(model)

  # get data
  dataset = MultiVideoDataset("../data/sim/train/", multi_frames=USE_RNN, combo=COMBO)
  train_split = int(len(dataset)*0.7) # 70% training data
  val_split = int(len(dataset)*0.3)   # 30% validation data
  train_set, val_set = random_split(dataset, [train_split+1, val_split])

  # loaders
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, collate_fn=CUSTOM_COLLATE, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, collate_fn=CUSTOM_COLLATE, pin_memory=True)

  # train model
  trainer = Trainer(device, model, train_loader, val_loader, model_path, writer_path, USE_RNN=USE_RNN, combo=COMBO)
  trainer.train(epochs=EPOCHS, lr=LR, use_mdn=USE_MDN)

  #dataset.cap.release()
  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
  torch.cuda.empty_cache()
