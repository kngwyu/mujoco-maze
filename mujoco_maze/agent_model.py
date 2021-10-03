"""Common APIs for defining mujoco robot.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle


class AgentModel(ABC, MujocoEnv, EzPickle):
    FILE: str
    MANUAL_COLLISION: bool
    ORI_IND: Optional[int] = None
    RADIUS: Optional[float] = None
    OBJBALL_TYPE: Optional[str] = None

    def __init__(self, file_path: str, frame_skip: int) -> None:
        MujocoEnv.__init__(self, file_path, frame_skip)
        EzPickle.__init__(self)

    def close(self):
        if self.viewer is not None and hasattr(self.viewer, "window"):
            import glfw

            glfw.destroy_window(self.viewer.window)
        super().close()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Returns the observation from the model."""
        pass

    def get_xy(self) -> np.ndarray:
        """Returns the coordinate of the agent."""
        pass

    def set_xy(self, xy: np.ndarray) -> None:
        """Set the coordinate of the agent."""
        pass
