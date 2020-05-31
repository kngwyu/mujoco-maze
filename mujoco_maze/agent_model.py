"""Common API definition for Ant and Point.
"""
from abc import ABC, abstractmethod
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle
from mujoco_py import MjSimState
import numpy as np


class AgentModel(ABC, MujocoEnv, EzPickle):
    FILE: str
    ORI_IND: int

    def __init__(self, file_path: str, frame_skip: int) -> None:
        MujocoEnv.__init__(self, file_path, frame_skip)
        EzPickle.__init__(self)

    def set_state_without_forward(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Returns the observation from the model.
        """
        pass

    @abstractmethod
    def get_xy(self) -> np.ndarray:
        """Returns the coordinate of the agent.
        """
        pass

    @abstractmethod
    def set_xy(self, xy: np.ndarray) -> None:
        """Set the coordinate of the agent.
        """
        pass

    @abstractmethod
    def get_ori(self) -> float:
        pass
