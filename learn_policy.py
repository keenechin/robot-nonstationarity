import gym
import numpy as np
from sklearn.neural_network import MLPRegressor
import time
import copy
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


def linear_softmax_policy(state, w):
    z = state.dot(w)
    exp = np.exp(z)
    return exp/np.sum(exp)


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


if __name__ == "__main__":
    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )
    env = gym.make('FiveBarPendulum-v0')
    np.random.seed(3)
    filename = 'hardware_model.sav'
    from os.path import exists
    if exists(filename):
        model = pickle.load(open(filename, 'rb'))
    else:
        model = learn_model(env, 1000)
        pickle.dump(model, open(filename, 'wb'))

    # Learning Code Here
    # REINFORCE

    start = time.time()
    NUM_EPISODES = 250
    LEARNING_RATE = 0.0025
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
            next_state = model.predict(np.reshape([*state[0], action], (1, -1)))
            reward = -env.cost(next_state[0])
            dsoftmax = softmax_grad(probs)[action, :]
            dlog = dsoftmax / probs[0, action]
            grad = state.T.dot(dlog[None, :])
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        for i in range(len(grads)):

            # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
            w += LEARNING_RATE * \
                grads[i] * sum([r * (GAMMA ** r)
                                for t, r in enumerate(rewards[i:])])

        # Append for logging and print
        episode_rewards.append(score)
        if score < best_score:
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
