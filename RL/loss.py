import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

def gaussian_probability(sigma, mu, target):
  target = target.unsqueeze(1).expand_as(sigma)
  ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
  return torch.prod(ret, 2)

# MTPLoss (mutliple-trajectory prediction loss), kinda like Mixture of Experts loss
# L2 Loss for each(i) path predicted
class MTPLoss:
  def __init__(self, n_modes, regression_loss_weight=1., angle_threshold_degrees=5., use_mdn=True):
    self.use_mdn = use_mdn

    self.n_modes = n_modes
    self.n_location_coords_predicted = 2  # (x,y) for each timestep
    self.trajectory_length = 200
    self.regression_loss_weight = regression_loss_weight
    self.angle_threshold = angle_threshold_degrees

  # splits the model predictions into mode probabilities and path
  def _get_trajectory_and_modes(self, model_pred):
    mode_probs = model_pred[:, -self.n_modes:].clone()
    desired_shape = (model_pred.shape[0], self.n_modes, -1, self.n_location_coords_predicted)

    if self.use_mdn:
      pi_len = self.trajectory_length*self.n_modes
      _len = self.trajectory_length * self.n_modes * 2
      pi = model_pred[:, :pi_len].reshape(desired_shape[:-1]).clone()
      sigma = model_pred[:, pi_len:(pi_len+_len)].clone().reshape(desired_shape)
      mean = model_pred[:, (pi_len+_len):-self.n_modes].clone().reshape(desired_shape)

      return mode_probs, pi, sigma, mean
    else:
      trajectories_no_modes = model_pred[:, :-self.n_modes].clone().reshape(desired_shape)
      return trajectories_no_modes, mode_probs

  # computes the angle between the last points of two paths (degrees)
  @staticmethod
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
  @staticmethod
  def _compute_ave_l2_norms(tensor):
    #l2_norms = torch.norm(tensor, p=2, dim=2)
    l2_norms = torch.norm(tensor, p=2, dim=1)
    avg_distance = torch.mean(l2_norms)
    return avg_distance.item()

  # compute angle between the target path and predicted paths
  def _compute_angles_from_ground_truth(self, target, trajectories):
    angles_from_ground_truth = []
    for mode, mode_trajectory in enumerate(trajectories):
        # For each mode, we compute the angle between the last point of the predicted trajectory for that
        # mode and the last point of the ground truth trajectory.
        #angle = self._angle_between(target[0], mode_trajectory)
        angle = self._angle_between(target, mode_trajectory)

        angles_from_ground_truth.append((angle, mode))
    return angles_from_ground_truth

  # finds the index of the best mode given the angles from the ground truth
  def _compute_best_mode(self, angles_from_ground_truth, target, trajectories):
    angles_from_ground_truth = sorted(angles_from_ground_truth)
    max_angle_below_thresh_idx = -1
    for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
      if angle <= self.angle_threshold:
        max_angle_below_thresh_idx = angle_idx
      else:
        break

    if max_angle_below_thresh_idx == -1:
      best_mode = random.randint(0, self.n_modes-1)
    else:
      distances_from_ground_truth = []
      for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx+1]:
        norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])
        distances_from_ground_truth.append((norm, mode))

      distances_from_ground_truth = sorted(distances_from_ground_truth)
      best_mode = distances_from_ground_truth[0][1]

    return best_mode

  # computes the MTP loss on a batch
  #The predictions are of shape [batch_size, n_ouput_neurons of last linear layer]
  #and the targets are of shape [batch_size, 1, n_timesteps, 2]
  def __call__(self, predictions, targets):
    batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
    if self.use_mdn:
      modes, pi, sigma, mean = self._get_trajectory_and_modes(predictions)
    else:
      trajectories, modes = self._get_trajectory_and_modes(predictions)

    for batch_idx in range(predictions.shape[0]):
      if self.use_mdn:
        angles = self._compute_angles_from_ground_truth(target=targets[batch_idx], trajectories=mean[batch_idx])
        best_mode = self._compute_best_mode(angles, target=targets[batch_idx], trajectories=mean[batch_idx])
        # best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)
        # regression_loss = F.smooth_l1_loss(best_mode_trajectory[0], targets[batch_idx])

        best_pi = pi[batch_idx, best_mode, :].unsqueeze(1)
        best_sigma = sigma[batch_idx, best_mode, :]
        best_mu = mean[batch_idx, best_mode, :]

        # regression_loss = mdn_loss(best_pi, best_sigma, best_mu, targets[batch_idx])
      else:
        angles = self._compute_angles_from_ground_truth(target=targets[batch_idx], trajectories=trajectories[batch_idx])
        best_mode = self._compute_best_mode(angles, target=targets[batch_idx], trajectories=trajectories[batch_idx])
        best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)
        regression_loss = F.smooth_l1_loss(best_mode_trajectory[0], targets[batch_idx])

      mode_probabilities = modes[batch_idx].unsqueeze(0)
      best_mode_target = torch.tensor([best_mode], device=predictions.device)
      classification_loss = F.cross_entropy(mode_probabilities, best_mode_target)
      
      loss = classification_loss + self.regression_loss_weight * regression_loss
      #deg = abs(math.atan(targets[batch_losses][0][-1][1]/targets[batch_idx][0][-1][0])*180/math.pi)
      #deg_wegith = math.exp(deg/20)
      #loss = loss * deg_weight
      batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)
    
    avg_loss = torch.mean(batch_losses)
    return avg_loss
