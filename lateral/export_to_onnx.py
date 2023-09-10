#!/usr/bin/env python3
import sys
import os
import torch.onnx
from model import *

"""
model_names = ["PathPlanner",
               "ComboModel",
               "SuperCombo"]
model_name = model_names[0]
"""

model_name = sys.argv[1]

model_path = "models/" + model_name + ".pth"
onnx_path = "models/" + model_name + ".onnx"

def export(model_path, onnx_path):
  input_names = ["road_image", "desire"]
  output_names = ["path"]
  #output_names = ["path", "crossroad"]

  model = PathPlanner() # CHANGE THIS
  """
  if model_name == model_names[0]:
    model = PathPlanner()
  elif model_name == model_names[1]:
    model = ComboModel()
  elif model_name == model_names[1]:
    model = SuperComboModel()
  else:
    print("Invalid model")
    exit(0)
  """
    
  model.load_state_dict(torch.load(model_path))
  model.eval()

  dummy_image = torch.randn(1, 3, 224, 224)
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

