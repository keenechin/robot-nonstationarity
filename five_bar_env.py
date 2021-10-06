from five_bar import FiveBar
from realsense_camera import RealsenseCamera
from dummy_camera import DummyCamera
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class FiveBarEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1, 1, 0, 0]),
            high=np.array([400, 600, 400, 600, 2*np.pi, 2*np.pi]),
            dtype=np.float64
        )
        self.camera = DummyCamera()
        # self.camera = RealsenseCamera()
        self.camera_feed = self.camera.get_feed()
        self.hardware = FiveBar()
        # self.camera.process.start()

    def _get_obs(self):
        servo_state = self.hardware.get_pos()
        pendulum_state = self.camera_feed.get()
        state = [*servo_state, *pendulum_state]
        return np.array(state)

    def step(self, u):
        self.hardware.primitive(u)
        state = self._get_obs()
        costs = self.reward(state)
        return state, -costs, False, {}

    def reward(self, state):
        return 0

    def reset(self):
        self.hardware.reset()
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":

    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )
    env = gym.make('FiveBarPendulum-v0')
    x0 = env.reset()

    env.hardware.test_motion()
