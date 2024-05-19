#!/usr/bin/env python3
import sys
import os
import torch.onnx
from model import *
from util import *

# USAGE: ./export_to_onnx.py ComboModel_01-10-23

"""
model_names = ["PathPlanner",
               "ComboModel",
               "SuperCombo"]
model_name = model_names[0]
"""

model_name = sys.argv[1]

# TODO: better way to process (full path + modify suffix)
model_path = "models/" + model_name + ".pt"
onnx_path = "models/" + model_name + ".onnx"

def export(model_path, onnx_path):
  input_names = ["road_images", "desire"]

  output_names = ["path"]
  model = PathPlannerRNN(hidden_size=HIDDEN_SIZE, n_layers=N_RNN_LAYERS)
  model.load_state_dict(torch.load(model_path))
  model.eval()

  dummy_image = torch.randn(1, N_FRAMES, 3, 224, 224)
  dummy_desire = torch.tensor([[1.0, 0.0, 0.0]])

  torch.onnx.export(
    model,
    (dummy_image, dummy_desire),
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    verbose=True
  )


if __name__ == "__main__":
  export(model_path, onnx_path)
