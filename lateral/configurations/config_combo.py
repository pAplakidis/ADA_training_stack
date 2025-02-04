from model.combo_model import ComboModel

# hyperparameters
BS = 128
EPOCHS = 10
LR = 1e-4

USE_MDN = False

model = ComboModel(use_mdn=USE_MDN)
