import gym
import numpy as np
from sklearn.neural_network import MLPRegressor
import time
import copy 

def test(env):
    num_drifts = 5
    for j in range(num_drifts):
        env.drift(env.hardware.linear_range *
                  (j / (num_drifts - 1)) + env.hardware.linear_min)
        for i in range(30):
            id = np.random.randint(0, 9)
            print(env.step(id))
        env.hardware.move_abs(
            env.hardware.theta1_mid, env.hardware.theta2_mid)
    env.reset()


def rollout(env, policy, num_steps):
    X = []
    y = []
    state = env.reset()
    for i in range(num_steps):
        action = policy(state)
        next_state, reward, _, _ = env.step(action)
        X.append([*state, action])
        y.append([*next_state])
        state = next_state
    return X, y


def random_policy(_):
    return np.random.randint(env.action_space.n)


def learn_model(env, N):
    model = MLPRegressor(max_iter=5000)
    X = []
    y = []
    X, y = rollout(env, random_policy, N)
    model.fit(X, y)
    print("Model Trained.")
    return model


def test_model(env, model, N):
    errors = np.empty((N, env.observation_space.shape[0]))
    X, y = rollout(env, random_policy, N)
    for i in range(N):
        expected_output = model.predict(np.reshape(X[i], (1, -1)))
        errors[i] = np.reshape(y[i], (1, -1)) - expected_output
    return errors


def policy(state, w):
    z = state.dot(w)
    exp = np.exp(z)
    return exp/np.sum(exp)


if __name__ == "__main__":
    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )
    env = gym.make('FiveBarPendulum-v0')
    start = time.time()
    model = learn_model(env, 100)
    errs = test_model(env, model, 100)
    print(f"Time passed: {time.time()-start}")
    env.reset()
    np.savetxt('errs.csv', errs, delimiter=',')
    env.camera.__del__()
