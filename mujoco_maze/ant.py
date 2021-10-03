"""
A four-legged robot as an explorer in the maze.
Based on `models`_ and `gym`_ (both ant and ant-v3).

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _gym: https://github.com/openai/gym
"""

from typing import Callable, Tuple

import numpy as np

from mujoco_maze.agent_model import AgentModel

ForwardRewardFn = Callable[[float, float], float]


def forward_reward_vabs(xy_velocity: float) -> float:
    return np.sum(np.abs(xy_velocity))


def forward_reward_vnorm(xy_velocity: float) -> float:
    return np.linalg.norm(xy_velocity)


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class AntEnv(AgentModel):
    FILE: str = "ant.xml"
    ORI_IND: int = 3
    MANUAL_COLLISION: bool = False
    OBJBALL_TYPE: str = "freejoint"

    def __init__(
        self,
        file_path: str,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        forward_reward_fn: ForwardRewardFn = forward_reward_vnorm,
    ) -> None:
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._forward_reward_fn = forward_reward_fn
        super().__init__(file_path, 5)

    def _forward_reward(self, xy_pos_before: np.ndarray) -> Tuple[float, np.ndarray]:
        xy_pos_after = self.sim.data.qpos[:2].copy()
        xy_velocity = (xy_pos_after - xy_pos_before) / self.dt
        return self._forward_reward_fn(xy_velocity)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        xy_pos_before = self.sim.data.qpos[:2].copy()
        self.do_simulation(action, self.frame_skip)

        forward_reward = self._forward_reward(xy_pos_before)
        ctrl_cost = self._ctrl_cost_weight * np.square(action).sum()

        return (
            self._get_obs(),
            self._forward_reward_weight * forward_reward - ctrl_cost,
            False,
            dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost),
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
            size=self.model.nq,
            low=-0.1,
            high=0.1,
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_ori(self) -> np.ndarray:
        ori = [0, 1, 0, 0]
        rot = self.sim.data.qpos[self.ORI_IND : self.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = np.arctan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.sim.data.qvel)

    def get_xy(self) -> np.ndarray:
        return np.copy(self.sim.data.qpos[:2])
