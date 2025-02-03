from model.path_planner import PathPlanner

PORTION = 1.0 # dataset percentage to use

SAVE_CHECKPOINTS = False

# hyperparameters
BS = 128
EPOCHS = 30
LR = 1e-4

N_WORKERS = 8
USE_MDN = False

model = PathPlanner(use_mdn=USE_MDN)
