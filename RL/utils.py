
# ENV
MAP_IDX = 3 # Town04
SYNC = True
STEP_TICKS = 10

# display image shape
IMG_WIDTH = 1164
IMG_HEIGHT = 874


SHOW_DISPLAY = False
EPISODE_LENGTH = 20

CRASH_REWARD = -200
SPEED_REWARD = -1
BASIC_REWARD = 1

# AGENT

# model constants
MODEL_NAME = "ADA"
W, H = 224, 224 # model image shape
MODEL_PATH = "models/path_planner_gru.pt"
N_FRAMES = 5
HIDDEN_SIZE = 500
N_GRU_LAYERS = 3

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

EPISODES = 100

DISCOUNT = 0.99
# TODO: might not be needed
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

