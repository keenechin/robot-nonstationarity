import gym
import numpy as np
from sklearn.neural_network import MLPRegressor
import time
import pickle


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
    nA = env.action_space.n

    for i in range(num_steps):
        probs = policy(state)
        action = np.random.choice(list(range(nA)), p=probs)
        next_state, reward, _, _ = env.step(action)
        X.append([*state, action])
        y.append([*next_state])
        state = next_state
    return X, y


def random_policy(state):
    return linear_softmax_policy(state, np.random.rand(10, 9))


def learn_model(env, policy, N):
    model = MLPRegressor(max_iter=5000)
    X = []
    y = []
    X, y = rollout(env, policy, N)
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


def linear_softmax_policy(state, params):
    z = state.dot(params)
    exp = np.exp(z)
    return exp/np.sum(exp)


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def learned_policy(state):
    params = np.array([[1.30599275,  0.8440695,  0.58461806, -0.02011946,  0.03461973,
                        0.54696131,  0.34817409,  0.09967214,  0.33051899],
                       [1.06756715,  0.91685071,  0.51103562,  0.44438496,  0.51257505,
                        0.59803864,  0.83532824,  0.47229371,  0.4040024],
                       [1.04607871,  0.84804166,  0.01283282,  0.66361759,  0.79730298,
                        0.67203189,  0.77656011,  0.44848724,  0.12165581],
                       [1.09589905,  0.51317097,  0.16997307,  0.68079135,  0.96937808,
                        0.22981028,  0.67614996,  0.484358,  0.71306881],
                       [1.08442556,  0.39196105,  0.21902421,  0.3574361,  0.73008704,
                        0.6766985,  0.46125652,  0.89495842,  0.44598672],
                       [0.49545462,  0.76035037,  0.0471648,  0.35860607,  0.76418505,
                        0.73295034,  0.30972206,  0.10925648,  0.46686322],
                       [0.35234971,  0.72541968,  0.1852073,  0.32916013,  0.84524875,
                        0.60527143,  0.89262316,  0.97511096,  0.83733929],
                       [0.20997242,  0.60701566,  0.47026797,  0.39273162,  0.74685098,
                        0.80762057,  0.6962998,  0.13834534,  0.70353977],
                       [0.06485459,  0.19487276,  0.92483907,  0.40695581,  0.14517594,
                        0.68020901,  0.15877936,  0.6467614,  0.25285785],
                       [0.05769313,  0.95822918,  0.06329934,  0.51090626,  0.34163173,
                        0.64786567,  0.84842912,  0.60230252,  0.58571582]])
    return linear_softmax_policy(state, params)


if __name__ == "__main__":
    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )
    env = gym.make('FiveBarPendulum-v0')
    np.random.seed(6)
    N = 252
    filename = f'hardware_model_{N}.sav'
    from os.path import exists
    if exists(filename):
        model = pickle.load(open(filename, 'rb'))
    else:
        model = learn_model(env, random_policy, N)
        pickle.dump(model, open(filename, 'wb'))

    # Learning Code Here
    # REINFORCE

    start = time.time()
    NUM_EPISODES = 253
    LEARNING_RATE = 0.000025
    GAMMA = 0.99
    nA = env.action_space.n
    episode_rewards = []

    w = np.random.rand(env.observation_space.shape[0], nA)
    best_w = w
    best_score = 0
    policy = linear_softmax_policy

    for e in range(NUM_EPISODES):
        state = np.reshape(env.reset(), (1, -1))
        grads = []
        rewards = []
        score = 0
        for i in range(100):
            probs = policy(state, w)
            action = np.random.choice(list(range(nA)), p=probs[0])
            next_state = model.predict(
                np.reshape([*state[0], action], (1, -1)))
            reward = -env.cost(next_state[0])
            dsoftmax = softmax_grad(probs)[action, :]
            dlog = dsoftmax / probs[0, action]
            grad = state.T.dot(dlog[None, :])
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        for i in range(len(grads)):

            # update towards the log policy gradient times **FUTURE** reward
            w += LEARNING_RATE * \
                grads[i] * sum([r * (GAMMA ** r)
                                for t, r in enumerate(rewards[i:])])

        # Append for logging and print
        episode_rewards.append(score)
        if score > best_score:
            best_w = w
            best_score = score
        print(f"EP: {str(e)} Score: {str(score)}")

    print((best_score, best_w))
    import matplotlib.pyplot as plt
    plt.plot(np.arange(NUM_EPISODES), episode_rewards)
    plt.show()
    # Wrapup code
    print(f"Time passed: {time.time()-start}")
    env.reset()
    env.camera.__del__()
