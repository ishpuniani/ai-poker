"""
We will make the agents play against each other to compare performance of each
"""
import collections
import json

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


def eval_agents(agent1_name, agent1, agent2_name, agent2):
    print('\n' + agent1_name + ' vs ' + agent2_name)
    env = rlcard.make('leduc-holdem')
    env.set_agents([agent1, agent2])
    reward_1, reward_2 = tournament(env, evaluate_num)
    print('Reward ' + agent1_name + ': ', reward_1)
    print('Reward ' + agent2_name + ': ', reward_2)
    return reward_1, reward_2


# Evaluate the performance.
# random_agent = RandomAgent(env.action_num)
# nfsp_model_path = os.path.join(rlcard.__path__[0], 'models/pretrained/leduc_holdem_nfsp')
nfsp_model_path = 'models/leduc_holdem_nfsp'
nfsp_agent = load_nfsp_leduc_agent(nfsp_model_path)
print("loaded NFSP leduc agent")

dqn_agent = load_dqn_leduc_agent('models/leduc_holdem_dqn')
print("loaded DQN leduc agent")

dqn_agent_rb = load_dqn_leduc_agent('models/leduc_holdem_dqn_rule_based')
print("loaded DQN RuleBased leduc agent")

cfr_agent = load_cfr_leduc_agent('models/cfr_model')
print("loaded CFR leduc agent")

cfr_agent_rb = load_cfr_leduc_agent('models/cfr_rule_based_model')
print("loaded CFR rule based leduc agent")

rule_based_agent = models.load('leduc-holdem-rule-v1').rule_agents[0]
print("loaded Leduc Rule Based Agent")

random_agent = RandomAgent(action_num=env.action_num)

# agents = [nfsp_agent, dqn_agent, dqn_agent_rb, cfr_agent, cfr_agent_rb, random_agent, rule_based_agent]
agents_dict = {
    "NFSP": nfsp_agent,
    "DQN": dqn_agent,
    "DQN Rule Based": dqn_agent_rb,
    "CFR": cfr_agent,
    "CFR Rule Based": cfr_agent_rb,
    "Random": random_agent,
    "Rule Based": rule_based_agent
}

results_dict = collections.defaultdict(dict)

agent_names = list(agents_dict.keys())

for i in range(len(agent_names)):
    for j in range(i, len(agent_names)):
        agent1_name = agent_names[i]
        agent2_name = agent_names[j]
        agent1 = agents_dict[agent1_name]
        agent2 = agents_dict[agent2_name]
        
        rew1, rew2 = eval_agents(agent1_name, agent1, agent2_name, agent2)
        results_dict[agent1_name][agent2_name] = rew1
        results_dict[agent2_name][agent1_name] = rew2

with open('eval.json', 'w') as fp:
    json.dump(results_dict, fp)
