# -*- coding: utf-8 -*-
# @File: test_resume.py
# @Author: Xiaocheng Tang
# @Date:   2019-05-12 19:59:05
from __future__ import absolute_import

import functools

from planet import training
from planet import tools
from planet.scripts import configs
from planet.training import trainer as trainer_
from planet import control
from environments.continuous_pricing_envs import UseScoreAsReward

import tensorflow as tf


# logdir = './logs/didi/00001'
# config = tools.AttrDict()
# params = tools.AttrDict({'logdir': './logs/didi'})
# with config.unlocked:
#     config = configs.debug(config, params)
# config = training.utility.load_config(logdir)


# logdir = './logs/didi/00001'
logdir = './logs/didi-default-v3/00001'
config = training.utility.load_config(logdir)

tf.reset_default_graph()

data = {'image': tf.placeholder(tf.float32, shape=(None, 10, 3, 3, 5), name='image'),
 'action': tf.placeholder(tf.float32, shape=(None, 10, 6), name='action'),
 'reward': tf.placeholder(tf.float32, shape=(None, 10), name='reward'),
 'score': tf.placeholder(tf.float32, shape=(None, 10), name='score'),
 'length': tf.placeholder(tf.int32, shape=(None,), name='length')}

trainer = trainer_.Trainer(logdir, config=config)

with tf.variable_scope('graph', use_resource=True):
    model_graph = dict()
    score, summary = training.define_model(data, trainer, config, model_graph)

model_graph = tools.AttrDict(model_graph)

for saver in config.savers:
    trainer.add_saver(**saver)

env_ctor = config.sim_collects['train-didi_pricing-cem-12'].task.env_ctor
env = env_ctor()
env = UseScoreAsReward(env)
env = control.wrappers.ConcatObservation(env, ['image'])
env = control.batch_env.BatchEnv([env], True)

agent_params = config.sim_collects['train-didi_pricing-cem-12']
planner = functools.partial(
    control.planning.cross_entropy_method,
    amount=5000,
    topk=20,
    iterations=20,
    horizon=12)
agent_config = tools.AttrDict(
    cell=model_graph.cell,
    encoder=model_graph.encoder,
    heads=model_graph.heads,
    planner=planner,
    objective=functools.partial(agent_params.objective, graph=model_graph),
    # exploration=agent_params.exploration,
    exploration=None,
    preprocess_fn=config.preprocess_fn,
    postprocess_fn=config.postprocess_fn)

# with agent_config.exploration.unlocked:
#     agent_config.exploration.scale = 0.1

agent = control.mpc_agent.MPCAgentSimple(env, agent_config)
observ = tf.placeholder(tf.float32, shape=(None, ) + env.observation_space.shape, name='observ')
agent_indices = tf.range(len(env))
begin_episode_op = agent.begin_episode(agent_indices)
perform_op = agent.perform(agent_indices, observ)
unroll_op = agent.unroll()

sess = trainer._create_session()
trainer._initialize_variables(
    sess, trainer._loaders, trainer._logdirs, trainer._checkpoints)

sess.run(begin_episode_op)

obs = env.reset()
action, _ = sess.run(perform_op, {observ: obs})
state = sess.run(unroll_op)
print(action, state)
import ipdb; ipdb.set_trace()

episodes = []
num_episodes = 1
for _ in range(num_episodes):
  policy = lambda obs: sess.run(perform_op, {observ: obs})[0]
  done = False
  obs = env.reset()
  total_reward = 0.0
  steps = 0
  while not done:
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    # print(action, reward)
    total_reward += reward[0]
    steps += 1
    # env.render()
  print(steps, total_reward)


