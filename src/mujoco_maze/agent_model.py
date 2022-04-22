"""Common APIs for defining mujoco robot.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from gym.utils.ezpickle import EzPickle

from mujoco_maze.our_mujoco_env import OurMujocoEnv


class AgentModel(ABC, OurMujocoEnv, EzPickle):
    FILE: str
    MANUAL_COLLISION: bool
    ORI_IND: Optional[int] = None
    RADIUS: Optional[float] = None
    OBJBALL_TYPE: Optional[str] = None

    def __init__(self, xml: Union[bytes, str], frame_skip: int) -> None:
        OurMujocoEnv.__init__(self, xml, frame_skip)
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
        raise NotImplementedError("get_xy is not implemented")

    def set_xy(self, xy: np.ndarray) -> None:
        """Set the coordinate of the agent."""
        raise NotImplementedError("set_xy is not implemented")
