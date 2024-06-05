# ENV
MAP_IDX = 3 # Town04
SYNC = True
STEP_TICKS = 10 # tick updates frames, so if we are running synchronous we need to skip some frames

# display image shape
IMG_WIDTH = 1164
IMG_HEIGHT = 874


SHOW_DISPLAY = False
EPISODE_LENGTH = 30

CRASH_REWARD = -200
LANE_INVASION_REWARD = -50
SPEED_REWARD = -1
BASIC_REWARD = 1

# AGENT

# model constants
MODEL_NAME = "ADA"
W, H = 224, 224 # model image shape
MODEL_PATH = "models/path_planner_gru.pt"
ACTOR_MODEL_SAVE_PATH = MODEL_PATH.split('.')[0] + "_rl.pt"
CRITIC_MODEL_SAVE_PATH = MODEL_PATH.split('.')[0] + "_rl_CRITIC.pt"
N_FRAMES = 5

# model hyperparameters
HIDDEN_SIZE = 500
N_GRU_LAYERS = 3
LR = 1e-4

# for path post-processing
N_MODES = 5 # number of trajectories from model output
N_LOCATION_COORDS_PREDICTED = 2
ANGLE_THRESHOLD = 5.
TRAJECTORY_LENGTH = 200 # 200 points in each path
N_COORDINATES = 2       # x, y

# RL constants
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5 # update target model every N episodes (TODO: might not be needed for policy gradient)
MEMORY_FRACTION = 0.8 # for GPU memory usage

EPISODES = 1000 # about 8-9 hrs

DISCOUNT = 0.99
# TODO: might not be needed
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

