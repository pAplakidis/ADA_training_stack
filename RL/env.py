#!/usr/bin/env python3
import os
import time
import math
import random
import numpy as np
import cv2
import carla

from utils import *
from carla_world_settings import *


class CarlaEnv:
  def __init__(self, carla_instance=None):
    self.client = carla.Client("localhost", 2000)
    self.client.set_timeout(2.0)
    print("[*] Loading Map:", maps[MAP_IDX])
    self.world = self.client.load_world(maps[MAP_IDX])
    self.bp_lib = self.world.get_blueprint_library()
    self.vehicle_bp = self.bp_lib.filter("model3")[0]
    self.vehicle = None

    self.frames_queue = []

  def reset(self):
    self.display_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    self.model_img = np.zeros((H, W, 3))

    self.collision_history = []
    self.actor_list = []

    # spawn car
    self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
    self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
    self.actor_list.append(self.vehicle)
    print("[*] Vehicle Spawned")

    # spawn camera
    camera_bp = self.bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
    camera_bp.set_attribute('fov', '70')
    camera_bp.set_attribute('sensor_tick', '0.05')
    spawn_point  = carla.Transform(carla.Location(x=0.8, z=1.13))  # dashcam location
    #spawn_point  = carla.Transform(carla.Location(x=-8., z=2.)) # NOTE: third-person camera view for debugging
    self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
    self.actor_list.append(self.camera)
    self.camera.listen(self.process_img)
    print("[*] Camera Spawned")

    # needed for vehicle initialization
    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
    time.sleep(4) # TODO: need to speed up episodes for training speed

    # spawn collision sensor
    self.collision_sensor_bp = self.bp_lib.find("sensor.other.collision")
    self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, spawn_point, attach_to=self.vehicle)
    self.actor_list.append(self.collision_sensor)
    self.collision_sensor.listen(lambda event: self.handle_collision_data(event))
    print("[*] Collision Sensor Spawned")

    while self.camera is None:
      time.sleep(0.01)

    # Enable synchronous mode
    if SYNC:
      settings = self.world.get_settings()
      settings.synchronous_mode = True 
      settings.no_rendering_mode = False
      settings.fixed_delta_seconds = 0.05
      self.world.apply_settings(settings)

      # cold start
      for _ in range(20):
        self.world.tick()

    # start episode
    self.episode_start = time.time()
    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

    return self.camera

  def step(self, steering_angle):
    # TODO: get action/control input from model (forward to model, get steering angle)
    self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steering_angle))

    v = self.vehicle.get_velocity()
    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

    # TODO: define values in utils
    # crashed
    if len(self.collision_history) != 0:
      done = True
      reward = CRASH_REWARD
      print("[-] Crashed")
    # prevent car from driving in circles
    # elif kmh < 50:
    #   done = False
    #   reward = SPEED_REWARD
    # reward for not colliding
    else:
      done = False
      reward = BASIC_REWARD

    if self.episode_start + EPISODE_LENGTH < time.time():
      done = True
      print("[*] Episode done")

    if SYNC:
      for i in range(STEP_TICKS):
        self.world.tick()

    # return next observation, reward, done, extra_info
    return self.frames_queue, reward, done, None

  def process_img(self, img):
    img = np.array(img.raw_data)
    img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    img = img[:, :, :3]
    self.display_img = img

    self.model_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    self.model_img = cv2.resize(self.model_img, (W,H))
    self.model_img = np.moveaxis(self.model_img, -1, 0)

    if len(self.frames_queue) >= N_FRAMES:
      del self.frames_queue[0]
    self.frames_queue.append(self.model_img)

  def handle_collision_data(self, event):
    self.collision_history.append(event)

  def destroy_agents(self):
    for actor in self.actor_list:
      actor.destroy()


if __name__ == "__main__":
  env = CarlaEnv()
  env.reset()

  try:
    idx = 0
    while True:
      camera, reward, done, _ = env.step(0.0)
      print(f"step {idx}")
      if SHOW_DISPLAY:
        cv2.imshow("Display IMG", env.display_img)
        cv2.waitKey(1)
      if done:
        break

      idx += 1
      time.sleep(1)
  except RuntimeError as re:
    print("[!]", re)
    print("Restarting ...")
  finally:
    env.destroy_agents()
    cv2.destroyAllWindows()
