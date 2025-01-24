from model.path_planner_rnn import PathPlannerRNN

PORTION = 1.0 # dataset percentage to use

# hyperparameters
BS = 20
EPOCHS = 200
LR = 1e-4

# model parameters
HIDDEN_SIZE = 512
N_RNN_LAYERS = 3

N_WORKERS = 8
USE_MDN = False

model = PathPlannerRNN(HIDDEN_SIZE, N_RNN_LAYERS)
