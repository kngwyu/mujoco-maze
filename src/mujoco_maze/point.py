"""
A ball-like robot as an explorer in the maze.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

from typing import Tuple

import mujoco
import numpy as np
from gym import spaces

from mujoco_maze.agent_model import AgentModel


class PointEnv(AgentModel):
    FILE: str = "point.xml"
    ORI_IND: int = 2
    MANUAL_COLLISION: bool = True
    RADIUS: float = 0.4
    OBJBALL_TYPE: str = "hinge"

    VELOCITY_LIMITS: float = 10.0

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path, 1)
        high = np.inf * np.ones(6, dtype=np.float32)
        high[3:] = self.VELOCITY_LIMITS * 1.2
        high[self.ORI_IND] = np.pi
        low = -high
        self.observation_space = spaces.Box(low, high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        qpos = self.data.qpos.copy()
        qpos[2] += action[1]
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        ori = qpos[2]
        # Compute increment in each direction
        qpos[0] += np.cos(ori) * action[0]
        qpos[1] += np.sin(ori) * action[0]
        qvel = np.clip(self.data.qvel, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        next_obs = self._get_obs()
        return next_obs, 0.0, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:3],  # Only point-relevant coords.
                self.data.qvel.flat[:3],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq,
            low=-0.1,
            high=0.1,
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_xy(self) -> np.ndarray:
        return self.data.qpos[:2].copy()

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.data.qvel)
