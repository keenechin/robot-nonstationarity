import gym
from os.path import exists
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


def rollout(env, policy, params, num_steps):
    X = []
    y = []
    state = env.reset()
    nA = env.action_space.n

    for i in range(num_steps):
        probs = policy(state, params)
        action = np.random.choice(list(range(nA)), p=probs)
        next_state, reward, _, _ = env.step(action)
        X.append([*state, action])
        y.append([*next_state])
        state = next_state
    return X, y


def learn_model(env, policy, params, N):
    model = MLPRegressor(max_iter=5000)
    X = []
    y = []
    X, y = rollout(env, policy, params, N)
    model.fit(X, y)
    print("Model Trained.")
    return model, X, y


def linear_softmax_policy(state, params):
    z = state.dot(params)
    exp = np.exp(z)
    return exp/np.sum(exp)


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def reinforce(policy, w, softmax_grad, env, model, NUM_EPISODES, LEARNING_RATE, LR_FINAL, GAMMA):
    print("Starting RL on trained dynamics model.")
    r = LR_FINAL ** (1/NUM_EPISODES)
    nA = env.action_space.n
    episode_rewards = []

    best_w = w
    best_score = 0

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
            reward = env.reward(next_state[0])
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

        # Learning rate decay
        LEARNING_RATE = LEARNING_RATE * r
        # Append for logging and print
        episode_rewards.append(score)
        if score > best_score:
            best_w = w
            best_score = score
        # print(LEARNING_RATE)
        print(f"Episode: {str(e)} Score: {str(score)}")
    return episode_rewards, best_w, best_score


def get_model_contingent(learn_model, linear_softmax_policy, env_name, env, N):
    random_collector = 'random'
    linear_collector = 'linear'
    random_data_dynamics_model = f'{env_name}_N{N}_{random_collector}policy'
    linear_data_dynamics_model = f'{env_name}_N{N}_{linear_collector}policy'
    random_data_collected = exists(random_data_dynamics_model)
    linear_data_collected = exists(linear_data_dynamics_model)
    random_policy_file = f"{random_data_dynamics_model}_trained_policy.sav"
    linear_policy_trained = exists(random_policy_file)
    linear_policy_file = f"{linear_data_dynamics_model}_trained_policy.sav"

    if linear_data_collected:
        print("Loading linear-trained dynamics model")
        w = np.random.rand(env.observation_space.shape[0], env.action_space.n)
        model = pickle.load(open(f"{linear_data_dynamics_model}.sav", 'wb'))
        policy_file = linear_policy_file
    elif linear_policy_trained:
        print("Training dynamics with linear policy")
        w = pickle.load(open(random_policy_file, 'rb'))
        model, X, y = learn_model(env, linear_softmax_policy, w, N)
        pickle.dump(model, open(f"{linear_data_dynamics_model}.sav", 'wb'))
        pickle.dump((X, y), open(f"{linear_data_dynamics_model}_data.sav", 'wb'))
        policy_file = linear_policy_file
    elif random_data_collected:
        print("Loading random-trained dynamics model")
        w = np.random.rand(env.observation_space.shape[0], env.action_space.n)
        model = pickle.load(open(f"{random_data_dynamics_model}.sav", 'rb'))
        policy_file = random_policy_file
    else:
        print("Training dynamics with random policy")
        w = np.random.rand(env.observation_space.shape[0], env.action_space.n)
        model, X, y = learn_model(env, linear_softmax_policy, w, N)
        pickle.dump(model, open(f"{random_data_dynamics_model}.sav", 'wb'))
        pickle.dump((X, y), open(f"{random_data_dynamics_model}_data.sav", 'wb'))
        policy_file = random_policy_file

    return policy_file, w, model


if __name__ == "__main__":
    gym.envs.register(
        id='FiveBarPendulum-v0',
        entry_point='five_bar_env:FiveBarEnv',
        kwargs={}
    )
    # env_name = 'CartPole-v1'
    env_name = 'FiveBarPendulum-v0'

    env = gym.make(env_name)
    np.random.seed(6)
    for i in range(8):
        N = 1000 * 5**(int(i/2))
        start = time.time()
        print(f"Collecting {N} samples.")
        policy_file, w, model = get_model_contingent(
            learn_model, linear_softmax_policy, env_name, env, N)
        print(f"Time passed to collect {N} samples: {(time.time()-start)/60.0} minutes")

        # Learning Code Here
        # REINFORCE

        start = time.time()
        NUM_EPISODES = 100
        LEARNING_RATE = 0.0001
        LR_FINAL = 0.2
        GAMMA = 0.99
        episode_rewards, best_w, best_score = reinforce(
            linear_softmax_policy, w, softmax_grad, env, model, NUM_EPISODES, LEARNING_RATE, LR_FINAL, GAMMA)

        # print((best_score, best_w))
        pickle.dump(best_w, open(policy_file, 'wb'))
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(NUM_EPISODES), episode_rewards)
        # plt.show()
        # Wrapup code
        print(f"Time passed for {NUM_EPISODES} RL Episodes: {(time.time()-start)/60} minutes")
        env.reset()

    env.camera.__del__()
