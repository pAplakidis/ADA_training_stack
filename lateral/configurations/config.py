from model.path_planner import PathPlanner

# hyperparameters
BS = 100
EPOCHS = 30
LR = 1e-4

USE_MDN = False

model = PathPlanner(use_mdn=USE_MDN)
