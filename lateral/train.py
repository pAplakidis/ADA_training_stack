#!/usr/bin/env python3
import os
import psutil
from torch.utils.data import DataLoader, random_split

from configurations.config import *
from util import *

from datasets.dataset_utils import *
from datasets.multi_video_dataset import MultiVideoDataset
from trainer import Trainer
from model.path_planner import PathPlanner
from model.combo_model import ComboModel
from model.path_planner_rnn import PathPlannerRNN
from model.super_combo_model import SuperComboModel

# EXAMPLE USAGE: 
# cp ./configurations/config_path_planner.py ./configurations/config.py && MODEL_PATH="models/path_planner.pt" WRITER_PATH="runs/test_1" ./train.py

# TODO: load model from checkpoint and resume training ("RESUME_FROM")

# change depending on hardware
n_cores = psutil.cpu_count(logical=False)
PREFETCH_FACTOR = n_cores
N_WORKERS = n_cores
EVAL_EPOCH = True
SAVE_CHECKPOINTS = True

model_path = os.getenv("MODEL_PATH")
if model_path == None:
  model_path = "models/path_planner_desire.pt"
print("[+] Model save path:", model_path)

PORTION = os.getenv("PORTION")
if PORTION == None:
  PORTION = 1.0
PORTION = float(PORTION)
print(f"[*] Using {PORTION * 100}% of dataset")

writer_path = os.getenv("WRITER_PATH")
print("[+] Tensorboard Writer path:", writer_path)


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[*] Using device:", device)

  print("\n[*] Configuration:")
  print("PORTION =", PORTION)
  print("BS =", BS)
  print("EPOCHS =", EPOCHS)
  print("LR =", LR)
  print("N_WORKERS =" ,N_WORKERS)
  print("USE_MDN =", USE_MDN)
  print()

  # define/select model
  if VERBOSE: print(model)

  # model selection variables
  if isinstance(model, PathPlanner):
    print("[+] Using model: PathPlanner")
    USE_RNN = False
    COMBO = False
    CUSTOM_COLLATE = custom_collate_pathplanner
  elif isinstance(model, PathPlannerRNN):
    USE_RNN = True
    COMBO = False
    CUSTOM_COLLATE = custom_collate_pathplanner_lstm
  elif isinstance(model, ComboModel):
    print("[+] Using model: ComboModel")
    USE_RNN = False
    COMBO = True
    CUSTOM_COLLATE = custom_collate_combomodel
  elif isinstance(model, SuperComboModel):
    print("[+] Using model: SuperComboModel")
    USE_RNN = True
    COMBO = True
    CUSTOM_COLLATE = custom_collate_combomodel
  else:
    print("Model not recognized!")
    exit(1)

  # get data
  dataset = MultiVideoDataset("../data/sim/train/",
                              multi_frames=USE_RNN, combo=COMBO, portion=PORTION, verbose=VERBOSE)
  train_split = int(len(dataset)*0.7) # 70% training data
  val_split = int(len(dataset)*0.3)   # 30% validation data
  train_set, val_set = random_split(dataset, [train_split+1, val_split])

  # loaders
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, prefetch_factor=PREFETCH_FACTOR,
                            num_workers=N_WORKERS, collate_fn=CUSTOM_COLLATE, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, prefetch_factor=PREFETCH_FACTOR,
                          num_workers=N_WORKERS, collate_fn=CUSTOM_COLLATE, pin_memory=True)

  # train model
  trainer = Trainer(device, model, train_loader, val_loader, model_path,
                    writer_path, eval_epoch=EVAL_EPOCH, use_rnn=USE_RNN,
                    use_mdn=USE_MDN, combo=COMBO, portion=PORTION, save_checkpoints=SAVE_CHECKPOINTS)
  trainer.train(epochs=EPOCHS, lr=LR)

  #dataset.cap.release()
  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
  torch.cuda.empty_cache()
