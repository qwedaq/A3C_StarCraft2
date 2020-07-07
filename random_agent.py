# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
import tensorflow as tf
import numpy as np
from utils import preprocess_minimap,preprocess_screen,minimap_channel,screen_channel
from network import network

class RandomAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""
    def __init__(self, minimap_size, screen_size, log):
        super(RandomAgent,self).__init__()
        self.temp_act_id = 0
        self.epsilon = [0.05, 0.2]
        self.action_size = len(actions.FUNCTIONS)
        self.minimap_size = minimap_size
        self.screen_size = screen_size
        self.summary = []
        self.buildNetwork()
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.counter = 0
        self.summary_writer = tf.summary.FileWriter(log,self.sess.graph)
  
    def buildNetwork(self):
    
        self.minimap = tf.placeholder(
            tf.float32,
            [None,minimap_channel(),self.minimap_size,self.minimap_size],
            name="minimap"
        )
        self.screen = tf.placeholder(
            tf.float32,
            [None, screen_channel(), self.screen_size,self.screen_size],
            name="screen"
        )
        self.info = tf.placeholder(
            tf.float32,
            [None,self.action_size],
            name="info"
        )
        self.spatial_action, self.non_spatial_action, self.value = network(self.minimap,self.screen,self.info,self.minimap_size,self.screen_size,self.action_size)
 # Set targets and masks
       
        self.valid_spatial_action = tf.placeholder(
            tf.float32, [None], name='valid_spatial_action')
        self.spatial_action_selected = tf.placeholder(
            tf.float32, [None, self.screen_size**2],
            name='spatial_action_selected')
        self.valid_non_spatial_action = tf.placeholder(
            tf.float32, [None, len(actions.FUNCTIONS)],
            name='valid_non_spatial_action')
        self.non_spatial_action_selected = tf.placeholder(
            tf.float32, [None, len(actions.FUNCTIONS)],
            name='non_spatial_action_selected')
        self.value_target = tf.placeholder(
            tf.float32, [None], name='value_target')
        # Compute log probability
        spatial_action_prob = tf.reduce_sum(
            self.spatial_action * self.spatial_action_selected, axis=1)
        spatial_action_log_prob = tf.log(
            tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
        non_spatial_action_prob = tf.reduce_sum(
            self.non_spatial_action * self.non_spatial_action_selected,
            axis=1)
        valid_non_spatial_action_prob = tf.reduce_sum(
            self.non_spatial_action * self.valid_non_spatial_action,
            axis=1)
        valid_non_spatial_action_prob = tf.clip_by_value(
            valid_non_spatial_action_prob, 1e-10, 1.)
        non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
        non_spatial_action_log_prob = tf.log(
            tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
        tf.summary.histogram('spatial_action_prob',
                                 spatial_action_prob)
        tf.summary.histogram('non_spatial_action_prob',non_spatial_action_prob)
        # Compute losses, more details in https://arxiv.org/abs/1602.01783
        # Policy loss and value loss
        action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
        advantage = tf.stop_gradient(self.value_target - self.value)
        policy_loss = -tf.reduce_mean(action_log_prob * advantage)
        value_loss = -tf.reduce_mean(self.value * advantage)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('value_loss', value_loss)
        # TODO: policy penalty
        self.reg = tf.squeeze(self.non_spatial_action)
        self.reg = tf.reduce_sum(self.reg*tf.log(self.reg))
        loss = policy_loss + value_loss + self.reg
        # Build the optimizer
        self.learning_rate = tf.placeholder(
            tf.float32, None, name='learning_rate')
        rmsprop = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate,decay=0.99, epsilon=1e-10)
        self.train_op = rmsprop.minimize(loss)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=100)

    def step(self, obs):
        super(RandomAgent, self).step(obs)
        self.randomOrgreedy = False
        feature_screen = np.expand_dims(preprocess_screen(obs.observation.feature_screen),axis=0)
        feature_map = np.expand_dims(preprocess_minimap(obs.observation.feature_minimap),axis=0)
        info = np.zeros([1, self.action_size], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1
        feed_dict = {self.minimap: feature_map, self.screen: feature_screen, self.info : info}
        non_spatial_action, spatial_action = self.sess.run([self.non_spatial_action, self.spatial_action], feed_dict=feed_dict)
        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel() #output shape 4096
        target = np.argmax(spatial_action) 
        target = [int(target // self.minimap_size), int(target % self.minimap_size)]
        valid_actions = obs.observation.available_actions
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        
        # print("available actions = " + str(obs.observation.available_actions))
        # function_id = numpy.random.choice(obs.observation.available_actions)
        # function_id = 1
        # print("function_id = " + str(function_id))
        # print("observation_spec " + str(self.obs_spec))
        # print("action_spec" + str((self.action_spec.functions)))
        # args = [[numpy.random.randint(0, size) for size in arg.sizes]
                # for arg in self.action_spec.functions[function_id].args]
        # print("function args = " + str(self.action_spec.functions[function_id].args))
        # for id in obs.observation.available_actions:
        #     for arg in self.action_spec.functions[id].args:
        #         ctr = 0
        #         for size in arg.sizes:
        #             ctr +=1
        #         if(ctr>2):
        #             print("function_id = " + str(id))
        
        if np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(valid_actions)
            self.randomOrgreedy = True
        if np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.screen_size - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.screen_size - 1, target[1] + dx)))
        act_args = []
        for arg in self.action_spec.functions[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful
        if(act_id != self.temp_act_id):
            self.temp_act_id = act_id
            if(self.randomOrgreedy):
                print("RANDOM")
            print("action " + str(actions.FUNCTIONS[act_id].name))
            print("target" + str(target))
        # print("args = " + str(args))
        # print("\n\n\n")
        return actions.FunctionCall(act_id, act_args)
        
    def update(self, rbs, disc, lr, cter):
        obs = rbs[-1][-1]
        if obs.last():
            R = 0
        else:
            feature_screen = np.expand_dims(preprocess_screen(obs.observation.feature_screen),axis=0)
            feature_map = np.expand_dims(preprocess_minimap(obs.observation.feature_minimap),axis=0)
            info = np.zeros([1, self.action_size], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1
            feed = {self.minimap: feature_map, self.screen: feature_screen, self.info : info}
            R = self.sess.run(self.value, feed_dict=feed)[0]

        # Compute targets and masks
        minimaps = []
        screens = []
        infos = []

        value_target = np.zeros([len(rbs)], dtype=np.float32)
        value_target[-1] = R

        valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
        spatial_action_selected = np.zeros(
            [len(rbs), self.screen_size**2], dtype=np.float32)
        valid_non_spatial_action = np.zeros(
            [len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros(
            [len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

        rbs.reverse()
        rewards = []
        for i, [obs, action, next_obs] in enumerate(rbs):
            feature_screen = np.expand_dims(preprocess_screen(obs.observation.feature_screen),axis=0)
            feature_map = np.expand_dims(preprocess_minimap(obs.observation.feature_minimap),axis=0)
            info = np.zeros([1, self.action_size], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1
            

            minimaps.append(feature_map)
            screens.append(feature_screen)
            infos.append(info)

            reward = obs.reward
            rewards.append(reward)
            act_id = action.function
            act_args = action.arguments

            value_target[i] = reward + disc * value_target[i - 1]

            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.screen_size + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1
        rewards = np.asarray(rewards,dtype=np.float32)
        reward_mean = np.mean(rewards)
        self.summary.append(tf.summary.scalar('rewards_mean', reward_mean))
        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        # Train
        feed = {
            self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr
        }
        _, summary = self.sess.run(
            [self.train_op,self.merged], feed_dict=feed)
        self.summary_writer.add_summary(summary, cter)
    
    def save_model(self):
        self.counter += 1
        self.saver.save(self.sess, "save_model" + '/model.pkl', self.counter)