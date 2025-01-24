from model.combo_model import ComboModel

SAVE_CHECKPOINTS = False

# hyperparameters
BS = 64
EPOCHS = 10
LR = 1e-4

N_WORKERS = 8
USE_MDN = False

model = ComboModel(use_mdn=USE_MDN)
