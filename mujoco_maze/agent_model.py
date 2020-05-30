"""Common API definition for Ant and Point.
"""
from abc import ABC, abstractmethod
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle
import numpy as np


class AgentModel(ABC, MujocoEnv, EzPickle):
    FILE: str
    ORI_IND: int

    def __init__(self, file_path: str, frame_skip: int) -> None:
        MujocoEnv.__init__(self, file_path, frame_skip)
        EzPickle.__init__(self)

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
