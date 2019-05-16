# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools


def encoder(obs):
  """Extract deterministic features from an observation."""
  kwargs = dict(activation=tf.nn.relu)
  hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())
  hidden = tf.layers.dense(hidden, 32, **kwargs)
  hidden = tf.layers.dense(hidden, 64, **kwargs)
  hidden = tf.layers.dense(hidden, 256, **kwargs)
  assert hidden.shape[1:].as_list() == [256], hidden.shape.as_list()
  hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
      np.prod(hidden.shape[1:].as_list())])
  return hidden


def decoder(state, data_shape):
  """Compute the data distribution of an observation from its state."""
  kwargs = dict(activation=tf.nn.relu)
  hidden = tf.layers.dense(state, 64, **kwargs)
  hidden = tf.reshape(hidden, [-1, hidden.shape[-1].value])
  hidden = tf.layers.dense(hidden, 32, **kwargs)
  mean = tf.layers.dense(hidden, 25, **kwargs)
  assert mean.shape[1:].as_list() == [25], mean.shape
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  # scale = tf.reshape(scale, tools.shape(state)[:-1] + data_shape)
  # scale = tf.maximum(tf.nn.softplus(scale), 0.5)
  # dist = tfd.Normal(mean, 1.0)
  scale = 0.5
  dist = tfd.Normal(mean, scale)
  # dist = tfd.Poisson(log_rate=mean)
  # dist = tfd.Exponential(tf.nn.softplus(mean))
  # dist = tfd.Gamma(tf.nn.softplus(mean), 1.0)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
