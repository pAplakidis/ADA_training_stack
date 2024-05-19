#!/usr/bin/env python3
import os
import time
import math
import random
import numpy as np
import cv2
import carla

from utils import *


class CarlaEnv:
  def __init__(self):
    self.client = carla.Client("localhost", 2000)
    self.client.set_timeout(2.0)
    self.world = self.client.get_world()
    self.bp_lib = self.world.get_blueprint_library()
    self.vehicle_bp = self.bp_lib.filter("model3")[0]
    self.vehicle = None

    self.reset()

  def reset(self):
    self.display_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    self.model_img = np.zeros((H, W, 3))

    self.collision_history = []
    self.actor_list = []

    # spawn car
    self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
    self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
    self.actor_list.append(self.vehicle)
    print("[+] Vehicle Spawned")

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
    # camera.listen(lambda img: car.process_img(img))
    # _camerad = Camerad(car)
    self.camera.listen(lambda img: self.process_img(img))
    print("[+] Camera Spawned")

    # needed for vehicle initialization
    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
    time.sleep(4) # TODO: need to speed up episodes for training speed

    # spawn collision sensor
    self.collision_sensor_bp = self.bp_lib.find("sensor.other.collision")
    self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, spawn_point, attach_to=self.vehicle)
    self.actor_list.append(self.collision_sensor)
    self.collision_sensor.listen(lambda event: self.handle_collision_data(event))
    print("[+] Collision Sensor Spawned")

    while self.camera is None:
      time.sleep(0.01)

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
      reward = -200
    # prevent car from driving in circles
    elif kmh < 50:
      done = False
      reward = -1
    # reward for not colliding forward
    else:
      done = False
      reward = 1

    if self.episode_start + EPISODE_LENGTH < time.time():
      done = True

    # return next observation, reward, done, extra_info
    return self.camera, reward, done, None

  def process_img(self, img):
    img = np.array(img.raw_data)
    img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    img = img[:, :, :3]

    self.display_img = img

    self.model_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    self.model_img = cv2.resize(self.model_img, (W,H))

    if SHOW_DISPLAY:
      cv2.imshow("Display IMG", self.display_img)
      cv2.imshow("Model IMG", self.model_img)
      cv2.waitKey(1)

  def handle_collision_data(self, event):
    self.collision_history.append(event)

  def destroy_agents(self):
    pass


if __name__ == "__main__":
  env = CarlaEnv()
  print("Hello")
