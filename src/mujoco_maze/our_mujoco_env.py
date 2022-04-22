from typing import Optional, Tuple, Union

import gym
import mujoco
import numpy as np
from gym import spaces

DEFAULT_SIZE = 500


class OurMujocoEnv(gym.Env):
    """MujocoEnv customized to use mujoco package"""

    def __init__(self, xml: Union[bytes, str], frame_skip: int = 1) -> None:
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)

    def _set_action_space(self) -> None:
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _set_observation_space(self, observation: np.ndarray) -> None:
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
        self.observation_space: spaces.Box = space

    def reset_model(self) -> np.ndarray:
        """Reset Mujoco model"""
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def _set_data(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        np.copyto(self.data.qpos, qpos)
        np.copyto(self.data.qvel, qvel)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self._set_data(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> int:
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        np.copyto(self.data.ctrl, ctrl)
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        raise NotImplementedError()

    def close(self) -> None:
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name: str) -> np.ndarray:
        # https://mujoco.readthedocs.io/en/latest/python.html#named-access
        return self.data.body(body_name).xpos

    def state_vector(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
