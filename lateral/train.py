#!/usr/bin/env python3
from torch.utils.data import DataLoader, random_split

from model import *
from dataset import *
from trainer import *
from util import *

# EXAMPLE USAGE: MODEL_PATH="models/path_planner.pt" WRITER_PATH="runs/test_1" ./train.py

model_path = os.getenv("MODEL_PATH")
if model_path == None:
  model_path = "models/path_planner_desire.pt"
print("[+] Model save path:", model_path)

writer_path = os.getenv("WRITER_PATH")
if writer_path == None:
  writer_path = "runs/train_eval_0"
print("[+] Tensorboard Writer path:", writer_path)

# HYPERPARAMETERS
BS = 2   # max Batch Size for current models on my PC
EPOCHS = 200
LR = 1e-5
HIDDEN_SIZE = 500
N_WORKERS = 8
N_GRU_LAYERS = 4
USE_MDN = False
VERBOSE = False


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # define/select model
  # model = PathPlanner(use_mdn=USE_MDN)
  # model = ComboModel(use_mdn=USE_MDN)
  # model = SuperComboModel(n_layers=N_GRU_LAYERS)
  model = PathPlannerLSTM(HIDDEN_SIZE)
  if VERBOSE: print(model)

  if isinstance(model, PathPlanner):
    print("[+] Using model: Pathplanner")
    use_rnn = False
    combo = False
    custom_collate = custom_collate_pathplanner
  elif isinstance(model, PathPlannerLSTM):
    use_rnn = True
    combo = False
    custom_collate = custom_collate_pathplanner_lstm
  elif isinstance(model, ComboModel):
    print("[+] Using model: ComboModel")
    use_rnn = False
    combo = True
    custom_collate = custom_collate_combomodel
  elif isinstance(model, SuperComboModel):
    print("[+] Using model: SuperComboModel")
    use_rnn = True
    combo = True
    custom_collate = custom_collate_combomodel
  else:
    print("Model not recognized!")
    exit(1)

  # get data
  dataset = MultiVideoDataset("../data/sim/train/", multi_frames=use_rnn, combo=combo)
  train_split = int(len(dataset)*0.7) # 70% training data
  val_split = int(len(dataset)*0.3)   # 30% validation data
  train_set, val_set = random_split(dataset, [train_split+1, val_split])
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, collate_fn=custom_collate, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, collate_fn=custom_collate, pin_memory=True)

  # train model
  trainer = Trainer(device, model, train_loader, val_loader, model_path, writer_path, use_rnn=use_rnn, combo=combo)
  trainer.train(epochs=EPOCHS, lr=LR, use_mdn=USE_MDN)

  #dataset.cap.release()
  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
