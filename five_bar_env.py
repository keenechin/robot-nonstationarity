import os
from five_bar import FiveBar
from realsense_camera import RealsenseCamera
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from sklearn.neural_network import MLPRegressor
import time


class FiveBarEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1, 1, 0, 0]),
            high=np.array([400, 600, 400, 600, 2*np.pi, 2*np.pi]),
            dtype=np.float64
        )
        self.camera = RealsenseCamera(viewport=[250, 85, 272, 172])
        self.camera.start()
        self.hardware = FiveBar()
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
        angle_from_goal = np.arctan2([state[4] - state[2]], [state[5] - state[3]])
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

    def test(self):
        num_drifts = 5
        for j in range(num_drifts):
            self.drift(self.hardware.linear_range *
                       (j / (num_drifts - 1)) + self.hardware.linear_min)
            for i in range(30):
                id = np.random.randint(0, 9)
                print(self.step(id))
            self.hardware.move_abs(
                self.hardware.theta1_mid, self.hardware.theta2_mid)
        self.reset()

    def idle(self, N):
        for i in range(N):
            print(self.step(self.stationaryid))

    def learn_model(self, x0, N):
        model = MLPRegressor()
        X = []
        y = []
        last_state = x0
        for i in range(N):
            action = np.random.randint(0, 9)
            state, reward, _, _ = self.step(action)
            X.append([*last_state, action])
            y.append([*state, reward])
            last_state = state
        model.fit(X, y)
        return model

    def test_model(self, model, N):
        expected_output = np.array([*self.reset(), 0])
        errors = np.empty((N, np.size(expected_output)))
        for i in range(N):
            action = np.random.randint(0, 9)
            state, reward, _, _ = self.step(action)
            errors[i] = np.abs(np.array([*state, reward]) - expected_output)
            expected_output = model.predict(np.reshape([*state, action], (1, -1)))
        return errors


if __name__ == "__main__":

    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )
    env = gym.make('FiveBarPendulum-v0')
    x0 = env.reset()
    print(x0)
    start = time.time()
    model = env.learn_model(x0, 100)
    errs = env.test_model(model, 100)
    print(f"Time passed: {time.time()-start}")
    env.reset()
    np.savetxt('errs.csv', errs, delimiter=',')
    env.camera.__del__()
