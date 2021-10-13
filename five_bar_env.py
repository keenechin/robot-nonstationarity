from five_bar import FiveBar
from realsense_camera import RealsenseCamera
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class FiveBarEnv(gym.Env):
    def __init__(self):
        self.camera = RealsenseCamera(viewport=[250, 85, 272, 172])
        self.camera.start()
        self.hardware = FiveBar()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=np.array([0, 0,
                          1, 1, 1, 1,
                          -np.pi, -np.pi, -np.pi, -np.pi]),
            high=np.array([2*np.pi, 2*np.pi,
                           self.camera.width, self.camera.height, self.camera.width, self.camera.height,
                           np.pi, np.pi, np.pi, np.pi]),
            dtype=np.float64
        )
        self.stationaryid = self.hardware.stationaryid

    def try_recover(self, keypoints):
        return [-1, -1, -1, -1, -1, -1, -1, -1]

    def _get_obs(self):
        servo_state = self.hardware.get_pos()
        success, pendulum_state = self.camera.feed.get()
        if success:
            state = [*servo_state, *pendulum_state]
        else:
            state = [*servo_state, *self.try_recover(pendulum_state)]
        return np.array(state)

    def step(self, u):
        self.hardware.primitive(u)
        state = self._get_obs()
        costs = self.cost(state)
        return state, -costs, False, {}

    def drift(self, pos):
        self.hardware.drift(pos)

    def cost(self, state):
        angle_from_goal = np.arctan2(
            [state[4] - state[2]], [state[5] - state[3]])
        return 1*angle_from_goal[0]

    def reset(self):
        self.hardware.reset()
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self.hardware.reset()
        self.camera.__del__()
