from datasets.dataset_utils import *
from model.path_planner import PathPlanner
from model.path_planner_rnn import PathPlannerRNN
from model.combo_model import ComboModel
from model.super_combo_model import SuperComboModel

# BS = 16 # combo model
BS = 4  # lstm model
EPOCHS = 200
LR = 1e-5

# HIDDEN_SIZE = 512 # TODO: retrain
HIDDEN_SIZE = 500
N_WORKERS = 8
N_RNN_LAYERS = 3

USE_MDN = False

model = PathPlanner(use_mdn=USE_MDN)
# model = ComboModel(use_mdn=USE_MDN)
# model = SuperComboModel(n_layers=N_GRU_LAYERS)
# model = PathPlannerRNN(HIDDEN_SIZE, N_RNN_LAYERS)

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

