import math
import random
import torch

from utils import *

 # path processing functions

# splits the model predictions into mode probabilities and path
def _get_trajectory_and_modes(model_pred):
  mode_probs = model_pred[:, -N_MODES:].clone()
  desired_shape = (model_pred.shape[0], N_MODES, -1, N_LOCATION_COORDS_PREDICTED)

  trajectories_no_modes = model_pred[:, :-N_MODES].clone().reshape(desired_shape)
  return trajectories_no_modes, mode_probs

# computes the angle between the last points of two paths (degrees)
def _angle_between(ref_traj, traj_to_compare):
  EPSILON = 1e-5

  if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
          ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
      raise ValueError('Both tensors should have shapes (-1, 2).')

  if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
      return 180. - EPSILON

  traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

  # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
  if math.isclose(traj_norms_product, 0):
      return 0.

  # We apply the max and min operations below to ensure there is no value
  # returned for cos_angle that is greater than 1 or less than -1.
  # This should never be the case, but the check is in place for cases where
  # we might encounter numerical instability.
  dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
  angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

  if angle >= 180:
      return angle - EPSILON

  return angle

# compute the average of l2 norms of each row in the tensor
def _compute_ave_l2_norms(tensor):
  #l2_norms = torch.norm(tensor, p=2, dim=2)
  l2_norms = torch.norm(tensor, p=2, dim=1)
  avg_distance = torch.mean(l2_norms)
  return avg_distance.item()

# compute angle between the target path and predicted paths
def _compute_angles_from_ground_truth(target, trajectories):
  angles_from_ground_truth = []
  for mode, mode_trajectory in enumerate(trajectories):
      # For each mode, we compute the angle between the last point of the predicted trajectory for that
      # mode and the last point of the ground truth trajectory.
      #angle = _angle_between(target[0], mode_trajectory)
      angle = _angle_between(target, mode_trajectory)

      angles_from_ground_truth.append((angle, mode))
  return angles_from_ground_truth

# finds the index of the best mode given the angles from the ground truth
def _compute_best_mode(angles_from_ground_truth, target, trajectories):
  angles_from_ground_truth = sorted(angles_from_ground_truth)
  max_angle_below_thresh_idx = -1
  for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
    if angle <= ANGLE_THRESHOLD:
      max_angle_below_thresh_idx = angle_idx
    else:
      break

  if max_angle_below_thresh_idx == -1:
    best_mode = random.randint(0, N_MODES-1)
  else:
    distances_from_ground_truth = []
    for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx+1]:
      norm = _compute_ave_l2_norms(target - trajectories[mode, :, :])
      distances_from_ground_truth.append((norm, mode))

    distances_from_ground_truth = sorted(distances_from_ground_truth)
    best_mode = distances_from_ground_truth[0][1]

  return best_mode

# very bad steering angle from path calculator
def calc_steering_angle(p0, p1):
  desired_theta =  90 - math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
  return desired_theta / 90
