#!/usr/bin/python
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
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
import threading

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "64",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.mark_flag_as_required("map")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("net", "atari", "atari or fcn.")
COUNTER = 0
# LOCK = threading.Lock()
def run_thread(agent_classes, players, map_name, visualize):
  """Run one thread worth of the environment with agents."""
  SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net
  LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
  if not os.path.exists(LOG):
    os.makedirs(LOG)
  with sc2_env.SC2Env(
      map_name=map_name,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_screen_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space,
          use_feature_units=FLAGS.use_feature_units),
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      disable_fog=FLAGS.disable_fog,
      visualize=False) as env:
    
    env = available_actions_printer.AvailableActionsPrinter(env)
    # wouldnt work for agent vs bot
    agents = [agent_cls(int(FLAGS.feature_minimap_size.x),int(FLAGS.feature_screen_size.x),LOG) for agent_cls in agent_classes]
    # run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
    replay_buffer = []
    for recorder, is_done in run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes):
      if FLAGS.training:
        
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          # with LOCK:
          global COUNTER
          COUNTER += 1
          counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          # print(replay_buffer)
          agents[0].update(replay_buffer, FLAGS.discount, learning_rate, counter)
          if counter % FLAGS.snapshot_step == 1:
            agents[0].save_model()
          replay_buffer = []
          # if counter % FLAGS.snapshot_step == 1:
          #   agents[0].save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
    if FLAGS.save_replay:
      env.save_replay(agent_classes[0].__name__)


def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  map_inst = maps.get(FLAGS.map)

  agent_classes = []
  players = []

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)
  agent_classes.append(agent_cls)
  players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                               FLAGS.agent_name or agent_name))

  if map_inst.players >= 2:
    if FLAGS.agent2 == "Bot":
      players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                 sc2_env.Difficulty[FLAGS.difficulty]))
    else:
      agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
      agent_cls = getattr(importlib.import_module(agent_module), agent_name)
      agent_classes.append(agent_cls)
      players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race],
                                   FLAGS.agent2_name or agent_name))

  # threads = []
  # for _ in range(FLAGS.parallel - 1):
  #   t = threading.Thread(target=run_thread,
  #                        args=(agent_classes, players, FLAGS.map, False))
  #   threads.append(t)
  #   t.start()

  run_thread(agent_classes, players, FLAGS.map, FLAGS.render)

  # for t in threads:
  #   t.join()

  # if FLAGS.profile:
  #   print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
