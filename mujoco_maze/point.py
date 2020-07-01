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
from typing import Optional, Tuple

import gym
import numpy as np

from mujoco_maze.agent_model import AgentModel


class PointEnv(AgentModel):
    FILE: str = "point.xml"
    ORI_IND: int = 2

    VELOCITY_LIMITS: float = 10.0

    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path, 1)
        high = np.inf * np.ones(6, dtype=np.float32)
        high[3:] = self.VELOCITY_LIMITS
        high[self.ORI_IND] = np.pi
        low = -high
        self.observation_space = gym.spaces.Box(low, high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        qpos = np.copy(self.sim.data.qpos)
        qpos[2] += action[1]
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        ori = qpos[2]
        # Compute increment in each direction
        qpos[0] += math.cos(ori) * action[0]
        qpos[1] += math.sin(ori) * action[0]
        qvel = np.clip(self.sim.data.qvel, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        return next_obs, 0.0, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:3],  # Only point-relevant coords.
                self.sim.data.qvel.flat[:3],
            ]
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
        return np.copy(self.sim.data.qpos[:2])

    def set_xy(self, xy):
        qpos = np.copy(self.sim.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.sim.data.qvel
        self.set_state(qpos, qvel)

    def get_ori(self):
        return self.sim.data.qpos[self.ORI_IND]
