# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for creating the ant environment in gym_mujoco."""

import math
import numpy as np

from mujoco_maze.agent_model import AgentModel


class PointEnv(AgentModel):
    FILE = "point.xml"
    ORI_IND = 2

    def __init__(self, file_path=None, expose_all_qpos=True):
        self._expose_all_qpos = expose_all_qpos
        super().__init__(file_path, 1)

    def _step(self, a):
        return self.step(a)

    def step(self, action):
        qpos = np.copy(self.sim.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -100, 100)
        qpos[1] = np.clip(qpos[1] + dy, -100, 100)
        qvel = self.sim.data.qvel
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = 0
        done = False
        info = {}
        return next_obs, reward, done, info

    def _get_obs(self):
        if self._expose_all_qpos:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[:3],  # Only point-relevant coords.
                    self.sim.data.qvel.flat[:3],
                ]
            )
        else:
            return np.concatenate(
                [self.sim.data.qpos.flat[2:3], self.sim.data.qvel.flat[:3]]
            )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.sim.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * 0.1

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_xy(self):
        qpos = self.sim.data.qpos
        return qpos[0], qpos[0]

    def set_xy(self, xy):
        qpos = np.copy(self.sim.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.sim.data.qvel
        self.set_state(qpos, qvel)

    def get_ori(self):
        return self.sim.data.qpos[self.ORI_IND]
