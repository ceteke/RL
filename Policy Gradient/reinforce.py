import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from time import time
from scipy.signal import medfilt

alpha = 0.01
discount = 0.99
episodes = 5000
report = 100

tf.reset_default_graph()
sess = tf.Session()

S_t = tf.placeholder(tf.float32, [None, 8])
expected_t = tf.placeholder(tf.float32, [None])
A_t = tf.placeholder(tf.int32, [None])
lr = tf.placeholder(tf.float32)

opt = tf.train.AdamOptimizer(lr)

logits = tf.layers.dense(S_t, 8, activation=tf.nn.tanh)
logits = tf.layers.dense(logits, 8, activation=tf.nn.tanh)
logits = tf.layers.dense(logits, 4)

action_prob = tf.nn.softmax(logits)
action_onehot = tf.one_hot(A_t, 4)
actions_log = tf.log(tf.reduce_sum(action_prob*action_onehot, axis=1))
loss = -tf.reduce_mean(actions_log * expected_t)

update_op = opt.minimize(loss)

def get_action(state):
    probs = sess.run(action_prob, feed_dict = {S_t: state})
    action = np.random.choice(range(len(probs.ravel())), p=probs.ravel())
    return action

env = gym.make('LunarLander-v2')
episode_rewards = []
losses = []
sess.run(tf.global_variables_initializer())

for e in range(episodes):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    start = time()
    # Get episode states, actions and rewards
    while not done:
        if e % report == 0:
            env.render()
        states.append(state)
        state = state.reshape(1, -1)
        action = get_action(state)
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)

    episode_rewards.append(np.sum(rewards))
    actions = np.array(actions)
    states = np.array(states)

    # Get expected discounted returns for each state
    expected_rewards = np.zeros_like(rewards)
    expected = 0
    for t in reversed(range(len(rewards))):
        expected = discount*expected + rewards[t]
        expected_rewards[t] = expected

    expected_rewards = np.array(expected_rewards)
    # print(expected_rewards)
    # Normalize expected discounted returns
    expected_rewards -= np.mean(expected_rewards)
    expected_rewards /= (np.std(expected_rewards) + 1e-8)

    if (e+1) % 1000 == 0:
        alpha /= 2

    l, _ = sess.run([loss, update_op], feed_dict={S_t:states,
                                                  expected_t:expected_rewards,
                                                  A_t: actions,
                                                  lr: alpha})
    losses.append(l)


    if e % report == 0:
        print("Episode {}, max reward: {}, {}".format(e, max(episode_rewards), alpha))
        if e >= 100:
            print("\tLast 100 mean: {}".format(np.mean(episode_rewards[e-100:e])))
    if e >= 100:
        if np.mean(episode_rewards[e-100:e]) >= 200:
            print("Solved at {}".format(e))
            break

filtered =  medfilt(np.array(episode_rewards, dtype=np.float32), kernel_size=101)
plt.plot(filtered)
plt.show()
