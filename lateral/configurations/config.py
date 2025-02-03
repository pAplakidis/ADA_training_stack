from model.combo_model import ComboModel

PORTION = 1.0 # dataset percentage to use

SAVE_CHECKPOINTS = False

# hyperparameters
BS = 128
EPOCHS = 10
LR = 1e-4

N_WORKERS = 8
USE_MDN = False

model = ComboModel(use_mdn=USE_MDN)
