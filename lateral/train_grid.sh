#!/bin/bash

# path planner

# 100% of dataset
echo -e "\n[++] Training path planner with 100% of dataset ...\n"
cp configurations/config_path_planner.py configurations/config.py && MODEL_PATH=models/path_planner.pth ./train.py

# ----------------------------

# combo

# 100% of dataset
echo -e "\n[++] Training combo model with 100% of dataset ...\n"
cp configurations/config_combo.py configurations/config.py && MODEL_PATH=models/combo_model.pth ./train.py
