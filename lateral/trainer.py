#!/usr/bin/env python3
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from util import *
from configurations.config import *
from model.model_utils import *
from loss.mtp_loss import MTPLoss
from loss.combo_loss import ComboLoss

# TODO: cleanup - different trainers/files for each model and dataset + switch from config
class Trainer:
  def __init__(self, device, model, train_loader, val_loader, model_path,
              writer_path=None, eval_epoch=False, use_rnn=False, use_mdn=False, combo=True, portion=1.0, save_checkpoints=False):
    self.use_rnn = use_rnn  # switch training RNN or CNN
    self.combo = combo      # switch PathPlanner or ComboModel/multitask
    self.use_mdn = use_mdn
    self.eval_epoch = eval_epoch
    self.model_path = model_path
    self.portion = portion
    self.save_checkpoints = save_checkpoints

    now = str(datetime.now())
    self.experiment_name = f"{now}-BS={BS}-LR={LR}-PORTION={self.portion}"
    self.model_dir = "/".join(model_path.split("/")[:-1])
    self.model_name = model_path.split("/")[-1].split(".")[0]

    if not writer_path:
      writer_path = os.path.join("runs", self.model_name, self.experiment_name)
    print("[TRAINER] Tensorboard output path:", writer_path)

    self.writer = SummaryWriter(writer_path)
    self.device = device
    print("[TRAINER] Device:", self.device)
    self.model = model.to(self.device)
    self.train_loader = train_loader
    self.val_loader = val_loader

  # TODO: implement resume_from_checkpoint
  def train(self, epochs=100, lr=1e-4):
    NANS = 0
    #loss_func = nn.MSELoss()
    if self.combo:
      loss_func = ComboLoss(2, self.model, self.device, use_mdn=self.use_mdn)
    else:
      loss_func = MTPLoss(self.model.n_paths, use_mdn=self.use_mdn)
    optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
    # scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.99)

    # evaluate model
    def eval(val_losses, train=False):
      if not train:
        print("[TRAINER] Evaluating ...")
      with torch.no_grad():
        try:
          self.model.eval()
          for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
            if self.use_rnn:
              IN_FRAMES = sample_batched["images"].float().to(self.device)
            else:
              X = torch.as_tensor(sample_batched["image"]).float().to(self.device)
            desire = torch.as_tensor(sample_batched["desire"]).float().to(self.device)
            Y_path = torch.as_tensor(sample_batched["path"]).float().to(self.device)
            if self.combo:
              Y_cr = torch.as_tensor(sample_batched["crossroad"]).float().to(self.device)

            if self.combo:
              if self.use_rnn:
                out_path, out_cr = self.model(IN_FRAMES, desire)
              else:
                out_path, out_cr = self.model(X, desire)
              loss = loss_func([out_path, out_cr], [Y_path, Y_cr])
            else:
              if self.use_rnn:
                out_path = self.model(IN_FRAMES, desire)
              else:
                out_path = self.model(X, desire)
              loss = loss_func(out_path, Y_path)

            if not self.combo and not torch.isnan(loss):
              if not train:
                self.writer.add_scalar('running evaluation loss', loss.item(), i_batch)
              val_losses.append(loss.item())
              t.set_description("Eval Batch Loss: %.2f"%(loss.item()))
            else:
              t.set_description(f"Eval Batch Loss: {loss[0].item():.2f} - path: {loss[1].item():.2f} - cr: {loss[2].item():.2f}")

        except KeyboardInterrupt:
          print("[~] Evaluation stopped by user")
      if not train:
        print("[TRAINER] Evaluation Done")
      return val_losses

    # train model
    losses = []
    vlosses = []
    try:
      print("[TRAINER] Training ...")
      idx = 0
      prev_min_vloss = math.inf
      for epoch in range(epochs):
        self.model.train()
        print("\n[=>] Epoch %d/%d"%(epoch+1, epochs))
        epoch_losses, epoch_path_plan_losses  = [], []
        epoch_vlosses = []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          if self.use_rnn:
            IN_FRAMES = sample_batched["images"].float().to(self.device)
          else:
            X = torch.as_tensor(sample_batched["image"]).float().to(self.device)
          desire = torch.as_tensor(sample_batched["desire"]).float().to(self.device)
          Y_path = torch.as_tensor(sample_batched["path"]).float().to(self.device)
          if self.combo:
            Y_cr = torch.as_tensor(sample_batched["crossroad"]).float().to(self.device)

          optim.zero_grad(set_to_none=True)
          if self.combo:
            if self.use_rnn:
              out_path, out_cr = self.model(IN_FRAMES, desire)
            else:
              out_path, out_cr = self.model(X, desire)
            loss, path_plan_loss, cr_loss = loss_func([out_path, out_cr], [Y_path, Y_cr])
          else:
            if self.use_rnn:
              out_path = self.model(IN_FRAMES, desire)
            else:
              out_path = self.model(X, desire)
            loss = loss_func(out_path, Y_path)

          if not torch.isnan(loss):
            self.writer.add_scalar("running loss", loss.item(), idx)
            epoch_losses.append(loss.item())
            if self.combo:
              self.writer.add_scalar('running path plan loss', path_plan_loss.item(), i_batch)
              epoch_path_plan_losses.append(path_plan_loss.item())

            loss.backward()
            optim.step()
          else:
            NANS += 1
            """
            print("NaN Loss Detected!")
            print(X)
            print(desire)
            print(Y_path)
            exit(0)
            """

          t.set_description("Batch Training Loss: %.2f"%(loss.item()))
          idx += 1

        avg_epoch_loss = np.array(epoch_losses).mean()
        losses.append(avg_epoch_loss)
        self.writer.add_scalar("epoch training loss", avg_epoch_loss, epoch)
        if self.combo:
          avg_epoch_path_plan_loss = np.array(epoch_path_plan_losses).mean()
          self.writer.add_scalar("epoch training PathPlanner loss", avg_epoch_path_plan_loss, epoch)
        print("[->] Epoch average training loss: %.4f"%(avg_epoch_loss))
        # scheduler.step()

        if self.eval_epoch:
          epoch_vlosses = eval(epoch_vlosses, train=True)
          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          vlosses.append(avg_epoch_vloss)
          # TODO: custom plot on the same figure as final training losses
          self.writer.add_scalar('epoch evaluation loss', avg_epoch_vloss, epoch)
          print("[->] Epoch average evaluation loss: %.4f"%(avg_epoch_vloss))

        if self.save_checkpoints and avg_epoch_vloss < prev_min_vloss:
          prev_min_vloss = avg_epoch_vloss
          save_checkpoint(
            os.path.join(self.model_dir, self.model_name, self.experiment_name, self.model_name + "_best.pt"),
            {
              "epoch": epoch,
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict": optim.state_dict(),
              "loss": avg_epoch_loss,
            }
          )

    except KeyboardInterrupt:
      print("[~] Training stopped by user")
    print("[TRAINER] Training Done")
    save_model(self.model_path, self.model)
    print("NaN losses detected:", NANS)

    # final evaluation
    val_losses = []
    val_losses = eval(val_losses)
    print("Avg Eval Loss: %.4f"%(np.array(val_losses).mean()))
  
    self.writer.close()
    return self.model
