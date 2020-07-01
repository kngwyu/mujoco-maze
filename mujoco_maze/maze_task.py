"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    THRESHOLD: float = 0.6

    def __init__(
        self, pos: np.ndarray, reward_scale: float = 1.0, rgb: Rgb = RED
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb

    def rbga_str(self) -> str:
        r, g, b = self.rgb
        return f"{r} {g} {b} 1"

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.THRESHOLD

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: float
    point: float


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 4.0)
    INNER_REWARD_SCALING: float = 0.0
    OBSERVE_BLOCKS: bool = False
    PUT_SPIN_NEAR_AGENT: bool = False

    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.goals = []

    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                return True
        return False

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class SingleGoalSparseUMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else -0.0001

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class SingleGoalDenseUMaze(SingleGoalSparseUMaze):
    REWARD_THRESHOLD: float = 1000.0

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs)


class SingleGoalSparsePush(SingleGoalSparseUMaze):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.375 * scale]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, E, R, B, B],
            [B, E, MazeCell.XY, E, B],
            [B, B, E, B, B],
            [B, B, B, B, B],
        ]


class SingleGoalDensePush(SingleGoalSparsePush):
    REWARD_THRESHOLD: float = 1000.0

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs)


class SingleGoalSparseFall(SingleGoalSparseUMaze):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 3.375 * scale, 4.5]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, C, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.CHASM, MazeCell.ROBOT
        return [
            [B, B, B, B],
            [B, R, E, B],
            [B, E, MazeCell.YZ, B],
            [B, C, C, B],
            [B, E, E, B],
            [B, B, B, B],
        ]


class SingleGoalDenseFall(SingleGoalSparseFall):
    REWARD_THRESHOLD: float = 1000.0

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs)


class SingleGoalSparse2Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 4.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return -0.0001

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B],
            [B, R, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, B],
            [B, B, B, B, B, E, B, B],
            [B, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B],
        ]


class SingleGoalDense2Rooms(SingleGoalSparse2Rooms):
    REWARD_THRESHOLD: float = 1000.0

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs)


class SubGoalSparse2Rooms(SingleGoalSparse2Rooms):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals.append(MazeGoal(np.array([5.0 * scale, 0.0 * scale]), 0.5, GREEN))


class SingleGoalSparse4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, 6.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return -0.0001

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, R, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, E, B, B, B, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class SingleGoalDense4Rooms(SingleGoalSparse4Rooms):
    REWARD_THRESHOLD: float = 1000.0

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs)


class SubGoalSparse4Rooms(SingleGoalSparse4Rooms):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals += [
            MazeGoal(np.array([0.0 * scale, 6.0 * scale]), 0.5, GREEN),
            MazeGoal(np.array([6.0 * scale, 0.0 * scale]), 0.5, GREEN),
        ]


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "UMaze": [SingleGoalDenseUMaze, SingleGoalSparseUMaze],
        "Push": [SingleGoalDensePush, SingleGoalSparsePush],
        "Fall": [SingleGoalDenseFall, SingleGoalSparseFall],
        "2Rooms": [SingleGoalDense2Rooms, SingleGoalSparse2Rooms, SubGoalSparse2Rooms],
        "4Rooms": [SingleGoalSparse4Rooms, SingleGoalDense4Rooms, SubGoalSparse4Rooms],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
