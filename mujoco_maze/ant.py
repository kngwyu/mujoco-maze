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
from typing import Tuple

import numpy as np

from mujoco_maze.agent_model import AgentModel


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class AntEnv(AgentModel):
    FILE = "ant.xml"
    ORI_IND = 3

    def __init__(self, file_path: Optional[str] = None) -> None:
        super().__init__(file_path, 5)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(action).sum()
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        _ = self.state_vector()
        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        # No cfrc observation
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:15],  # Ensures only ant obs.
                self.sim.data.qvel.flat[:14],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_ori(self):
        ori = [0, 1, 0, 0]
        ori_ind = self.ORI_IND
        rot = self.sim.data.qpos[ori_ind : ori_ind + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy):
        qpos = np.copy(self.sim.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.sim.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return np.copy(self.sim.data.qpos[:2])
