"""
We will make the agents play against each other to compare performance of each
"""

import tensorflow as tf
import os

import rlcard
from rlcard import models
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents.cfr_agent import CFRAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament

# Make environment
env = rlcard.make('leduc-holdem')

# The initial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# Number of games to be played
evaluate_num = 10000


def load_nfsp_leduc_agent(model_path):
    # Set a global seed
    set_global_seed(0)

    # Load pretrained model
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        nfsp_agents = []
        for i in range(env.player_num):
            agent = NFSPAgent(sess,
                              scope='nfsp' + str(i),
                              action_num=env.action_num,
                              state_shape=env.state_shape,
                              hidden_layers_sizes=[128,128],
                              q_mlp_layers=[128,128])
            nfsp_agents.append(agent)

    # We have a pretrained model here. Change the path for your model.
    # check_point_path = os.path.join(rlcard.__path__[0], 'models/pretrained/leduc_holdem_nfsp')
    check_point_path = model_path

    with sess.as_default():
        with graph.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))

    return nfsp_agents[0]


def load_dqn_leduc_agent(model_path):
    # Set a global seed
    set_global_seed(0)

    # Load pretrained model
    # tf.reset_default_graph()
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        nfsp_agents = []
        agent = DQNAgent(sess,
                         scope='dqn',
                         action_num=env.action_num,
                         replay_memory_init_size=memory_init_size,
                         train_every=train_every,
                         state_shape=env.state_shape,
                         mlp_layers=[128, 128])

    # We have a pretrained model here. Change the path for your model.
    # check_point_path = os.path.join(rlcard.__path__[0], 'models/pretrained/leduc_holdem_nfsp')
    check_point_path = model_path

    with sess.as_default():
        with graph.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
    return agent


def load_cfr_leduc_agent(model_path):
    agent = CFRAgent(env, model_path=model_path)
    agent.load()
    return agent


# Evaluate the performance.
# random_agent = RandomAgent(env.action_num)
# nfsp_model_path = os.path.join(rlcard.__path__[0], 'models/pretrained/leduc_holdem_nfsp')
nfsp_model_path = 'models/leduc_holdem_nfsp'
nfsp_agent = load_nfsp_leduc_agent(nfsp_model_path)
print("loaded NFSP leduc agent")

dqn_agent = load_dqn_leduc_agent('models/leduc_holdem_dqn')
print("loaded DQN leduc agent")

cfr_agent = load_cfr_leduc_agent('models/cfr_model')
print("loaded CFR leduc agent")

rule_based_agent = models.load('leduc-holdem-rule-v2')
print("loaded Leduc Rule Based Agent")

print('\nNFSP vs DQN')
env1 = rlcard.make('leduc-holdem')
env1.set_agents([nfsp_agent, dqn_agent])
reward_nfsp, reward_dqn = tournament(env1, evaluate_num)
print('Reward NFSP: ', reward_nfsp)
print('Reward DQN: ', reward_dqn)

print('\nCFR vs DQN')
env2 = rlcard.make('leduc-holdem')
env2.set_agents([cfr_agent, dqn_agent])
reward_cfr, reward_dqn = tournament(env2, evaluate_num)
print('Reward CFR: ', reward_cfr)
print('Reward DQN: ', reward_dqn)

print('\nCFR vs NFSP')
env3 = rlcard.make('leduc-holdem')
env3.set_agents([cfr_agent, nfsp_agent])
reward_cfr, reward_nfsp = tournament(env3, evaluate_num)
print('Reward CFR: ', reward_cfr)
print('Reward NFSP: ', reward_nfsp)
