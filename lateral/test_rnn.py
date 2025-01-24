#!/usr/bin/env python3
import os
import io
import time
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import torch

from util import *
from lateral.configurations.config import *
from datasets.dataset_utils import *
from datasets.multi_video_dataset import MultiVideoDataset
from model.path_planner_rnn import PathPlannerRNN
from model.model_utils import *
from loss.mtp_loss import MTPLoss

# EXAMPLE USAGE: DATADIR="../data/sim/test/29/" MODEL_PATH="models/ComboModel.pt" ./test_video.py

base_dir = os.getenv("DATADIR")
if base_dir is None:
  base_dir = "../data/sim/train/22/"
  print("[~] No DATA directory specified, using default:", base_dir)

model_path = os.getenv("MODEL_PATH")
if model_path == None:
  model_path = "models/path-planner-lstm.pt"
print("[+] Model save path:", model_path)

def figshow(fig):
  buf = io.BytesIO()
  pio.write_image(fig, buf)
  buf.seek(0)
  file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  cv2.imshow("Predicted Path", img)


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # get data
  cap = cv2.VideoCapture(base_dir+"video.mp4")
  path_dir = base_dir + "frame_paths.npy"
  cr_dir = base_dir+"crossroads.npy"
  desire_dir = base_dir + "desires.npy"
  ground_truth = False

  if os.path.isfile(path_dir):
    ground_truth = True
    frame_paths = np.load(path_dir)
    if os.path.isfile(cr_dir):
      crossroads = np.load(cr_dir)

  desires = one_hot_encode(np.load(desire_dir))

  model = PathPlannerRNN(HIDDEN_SIZE, N_RNN_LAYERS).to(device)
  model = load_model(model_path, model)
  model.eval()
  # print(model)

  loss_func = MTPLoss(model.n_paths, use_mdn=False)
  fig = go.FigureWidget()

  fid = 0 # frame id/index
  prev_frame_time = 0
  new_frame_time = 0
  frames = []

  # FIXME: desires, ground-truth and frames are not synced properly
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    print("[+] Frame:", fid)
    
    # get N_FRAMES consecutive frames
    frames.append(np.moveaxis(cv2.resize(frame, (W,H)), -1, 0))
    if fid < N_FRAMES-1:
      fid += 1
      continue

    # preprocess input
    IN_FRAMES = torch.as_tensor(np.array([frames, frames])).float().to(device)

    if ground_truth:
      path = frame_paths[fid]
      print("GroundTruth Path shape:", path.shape)
      if np.isnan(path).any():
        path = np.zeros_like(path)
    desire = desires[fid]
    desire_idx = np.argmax(desire)
    print("Desire:", desire_idx, "=>", DESIRE[desire_idx])

    disp_img = cv2.resize(frame, (d_W,d_H))

    # forward to model
    with torch.no_grad():
      DES = torch.as_tensor(np.array([desire, desire])).float().to(device)
      print("Model input shape:", IN_FRAMES.shape)

      out_path = model(IN_FRAMES, DES)
      print("Model output shape:", out_path.shape)
      trajectories, modes = loss_func._get_trajectory_and_modes(out_path)

      # process model outputs
      print("Path probabilities:")
      for i in range(len(modes[0])):
        print("%d => %.2f" % (i, modes[0][i].item()))

      for idx, pred_path in enumerate(trajectories[0]):
        path_x = pred_path.to("cpu").numpy()[:, 0]
        path_y = pred_path.to("cpu").numpy()[:, 1]
        if modes[0][idx] == torch.max(modes[0]):
          marker = {"color": "red"}
          name = "%.2f - best_path"%modes[0][idx].item()
        else:
          marker = {"color": "blue"}
          name = "%.2f - path%d"%(modes[0][idx].item(), idx)
        fig.add_scatter(x=path_x, y=path_y, name=name, marker=marker)

    # display ground-truth
    if ground_truth:
      fig.add_scatter(x=path[:, 0], y=path[:, 1], name="ground_truth", marker={"color": "green"})
    fig.update_layout(xaxis_range=[-50,50])
    fig.update_layout(yaxis_range=[0,50])
    figshow(fig)
    fig.data = []

    # Image Display
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    thickness = 2

    # display desire
    org = (25, 55)
    color = (255, 0, 0)
    text = "DESIRE: %s, %d" % (DESIRE[desire_idx], desire_idx)
    disp_img = cv2.putText(disp_img, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

    # display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    print("FPS:", fps)
    disp_img = cv2.putText(disp_img ,"FPS:"+fps, (d_W - 150, 25), font,
                      fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    # finish loop
    cv2.imshow("DISPLAY", disp_img)
    cv2.waitKey(1)
    fid += 1
    del frames[0]
    print()

cap.release()
cv2.destroyAllWindows()

