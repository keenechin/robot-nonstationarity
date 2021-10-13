import gym
import numpy as np
from sklearn.neural_network import MLPRegressor
import time


if __name__ == "__main__":
    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )

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

    def idle(env, N):
        for i in range(N):
            print(env.step(env.stationaryid))

    def learn_model(env, x0, N):
        model = MLPRegressor(max_iter=5000)
        X = []
        y = []
        last_state = x0
        for i in range(N):
            action = np.random.randint(0, 9)
            state, reward, _, _ = env.step(action)
            X.append([*last_state, action])
            y.append([*state, reward])
            last_state = state
        model.fit(X, y)
        print("Model Trained.")
        return model

    def test_model(env, model, N):
        expected_output = np.array([*env.reset(), 0])
        errors = np.empty((N, np.size(expected_output)))
        for i in range(N):
            action = np.random.randint(0, 9)
            state, reward, _, _ = env.step(action)
            errors[i] = np.abs(np.array([*state, reward]) - expected_output)
            expected_output = model.predict(np.reshape([*state, action], (1, -1)))
        return errors

    env = gym.make('FiveBarPendulum-v0')
    x0 = env.reset()
    start = time.time()
    model = learn_model(env, x0, 100)
    errs = test_model(env, model, 100)
    print(f"Time passed: {time.time()-start}")
    env.reset()
    np.savetxt('errs.csv', errs, delimiter=',')
    env.camera.__del__()