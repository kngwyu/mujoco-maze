"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Tuple, Type

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
    INNER_REWARD_SCALING: float = 0.01
    TOP_DOWN_VIEW: bool = False
    OBSERVE_BLOCKS: bool = False
    PUT_SPIN_NEAR_AGENT: bool = False

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale

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


class DistRewardMixIn:
    REWARD_THRESHOLD: float = -1000.0
    goals: List[MazeGoal]
    scale: float

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs) / self.scale


class GoalRewardUMaze(MazeTask):
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


class DistRewardUMaze(GoalRewardUMaze, DistRewardMixIn):
    pass


class GoalRewardPush(GoalRewardUMaze):
    TOP_DOWN_VIEW = True

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


class DistRewardPush(GoalRewardPush, DistRewardMixIn):
    pass


class GoalRewardFall(GoalRewardUMaze):
    TOP_DOWN_VIEW = True

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


class DistRewardFall(GoalRewardFall, DistRewardMixIn):
    pass


class GoalReward2Rooms(MazeTask):
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


class DistReward2Rooms(GoalReward2Rooms, DistRewardMixIn):
    pass


class SubGoal2Rooms(GoalReward2Rooms):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals.append(MazeGoal(np.array([5.0 * scale, 0.0 * scale]), 0.5, GREEN))


class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]

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
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, E, B, B, B, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class DistReward4Rooms(GoalReward4Rooms, DistRewardMixIn):
    pass


class SubGoal4Rooms(GoalReward4Rooms):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals += [
            MazeGoal(np.array([0.0 * scale, -6.0 * scale]), 0.5, GREEN),
            MazeGoal(np.array([6.0 * scale, 0.0 * scale]), 0.5, GREEN),
        ]


class GoalRewardTRoom(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0)

    def __init__(
        self,
        scale: float,
        goals: List[Tuple[float, float]] = [(2.0, -3.0)],
    ) -> None:
        super().__init__(scale)
        self.goals = []
        for x, y in goals:
            self.goals.append(MazeGoal(np.array([x * scale, y * scale])))

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return -0.0001

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, B, E, E, B],
            [B, E, E, B, E, E, B],
            [B, E, B, B, B, E, B],
            [B, E, E, R, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistRewardTRoom(GoalRewardTRoom, DistRewardMixIn):
    pass


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "UMaze": [DistRewardUMaze, GoalRewardUMaze],
        "Push": [DistRewardPush, GoalRewardPush],
        "Fall": [DistRewardFall, GoalRewardFall],
        "2Rooms": [DistReward2Rooms, GoalReward2Rooms, SubGoal2Rooms],
        "4Rooms": [DistReward4Rooms, GoalReward4Rooms, SubGoal4Rooms],
        "TRoom": [DistRewardTRoom, GoalRewardTRoom],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
