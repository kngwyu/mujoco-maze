"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 0.6,
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 4.0, 4.0)
    INNER_REWARD_SCALING: float = 0.01
    # For Fall/Push/BlockMaze
    OBSERVE_BLOCKS: bool = False
    # For Billiard
    OBSERVE_BALLS: bool = False
    OBJECT_BALL_SIZE: float = 1.0
    # Unused now
    PUT_SPIN_NEAR_AGENT: bool = False
    TOP_DOWN_VIEW: bool = False

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
    PENALTY: float = -0.0001

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

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


class GoalRewardSimpleRoom(GoalRewardUMaze):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([2.0 * scale, 0.0]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, B, B],
        ]


class DistRewardSimpleRoom(GoalRewardSimpleRoom, DistRewardMixIn):
    pass


class GoalRewardPush(GoalRewardUMaze):
    OBSERVE_BLOCKS: bool = True

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.375 * scale]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R, M = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT, MazeCell.XY_BLOCK
        return [
            [B, B, B, B, B],
            [B, E, R, B, B],
            [B, E, M, E, B],
            [B, B, E, B, B],
            [B, B, B, B, B],
        ]


class DistRewardPush(GoalRewardPush, DistRewardMixIn):
    pass


class GoalRewardFall(GoalRewardUMaze):
    OBSERVE_BLOCKS: bool = True

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 3.375 * scale, 4.5]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, C, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.CHASM, MazeCell.ROBOT
        M = MazeCell.YZ_BLOCK
        return [
            [B, B, B, B],
            [B, R, E, B],
            [B, E, M, B],
            [B, C, C, B],
            [B, E, E, B],
            [B, B, B, B],
        ]


class DistRewardFall(GoalRewardFall, DistRewardMixIn):
    pass


class GoalReward2Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float, goal: Tuple[int, int] = (4.0, -2.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, B],
            [B, E, E, E, B, E, E, B],
            [B, E, R, E, B, E, E, B],
            [B, E, E, E, B, E, E, B],
            [B, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B],
        ]


class DistReward2Rooms(GoalReward2Rooms, DistRewardMixIn):
    pass


class SubGoal2Rooms(GoalReward2Rooms):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (4.0, -2.0),
        subgoals: List[Tuple[float, float]] = [(1.0, -2.0), (-1.0, 2.0)],
    ) -> None:
        super().__init__(scale, primary_goal)
        for subgoal in subgoals:
            self.goals.append(
                MazeGoal(np.array(subgoal) * scale, reward_scale=0.5, rgb=GREEN)
            )


class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

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
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float, goal: Tuple[float, float] = (2.0, -3.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

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


class SubGoalTRoom(GoalRewardTRoom):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (2.0, -3.0),
        subgoal: Tuple[float, float] = (-2.0, -3.0),
    ) -> None:
        super().__init__(scale, primary_goal)
        self.goals.append(
            MazeGoal(np.array(subgoal) * scale, reward_scale=0.5, rgb=GREEN)
        )


class GoalRewardBlockMaze(GoalRewardUMaze):
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 4.0, None)
    OBSERVE_BLOCKS: bool = True

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 3.0 * scale]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        M = MazeCell.XY_BLOCK
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, M, B],
            [B, E, E, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class DistRewardBlockMaze(GoalRewardBlockMaze, DistRewardMixIn):
    pass


class GoalRewardBilliard(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(None, 3.0, None)
    OBSERVE_BALLS: bool = True
    GOAL_SIZE: float = 0.3

    def __init__(self, scale: float, goal: Tuple[float, float] = (2.0, -3.0)) -> None:
        super().__init__(scale)
        goal = np.array(goal) * scale
        self.goals.append(
            MazeGoal(goal, threshold=self._threshold(), custom_size=self.GOAL_SIZE)
        )

    def _threshold(self) -> float:
        return self.OBJECT_BALL_SIZE + self.GOAL_SIZE

    def reward(self, obs: np.ndarray) -> float:
        object_pos = obs[3:6]
        for goal in self.goals:
            if goal.neighbor(object_pos):
                return goal.reward_scale
        return self.PENALTY

    def termination(self, obs: np.ndarray) -> bool:
        object_pos = obs[3:6]
        for goal in self.goals:
            if goal.neighbor(object_pos):
                return True
        return False

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B = MazeCell.EMPTY, MazeCell.BLOCK
        R, M = MazeCell.ROBOT, MazeCell.OBJECT_BALL
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, E, E, M, E, E, B],
            [B, E, E, R, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistRewardBilliard(GoalRewardBilliard):
    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs[3:6]) / self.scale


class SubGoalBilliard(GoalRewardBilliard):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (2.0, -3.0),
        subgoals: List[Tuple[float, float]] = [(-2.0, -3.0), (-2.0, 1.0), (2.0, 1.0)],
    ) -> None:
        super().__init__(scale, primary_goal)
        for subgoal in subgoals:
            self.goals.append(
                MazeGoal(
                    np.array(subgoal) * scale,
                    reward_scale=0.5,
                    rgb=GREEN,
                    threshold=self._threshold(),
                    custom_size=self.GOAL_SIZE,
                )
            )


class BanditBilliard(SubGoalBilliard):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (4.0, -2.0),
        subgoals: List[Tuple[float, float]] = [(4.0, 2.0)],
    ) -> None:
        super().__init__(scale, primary_goal, subgoals)

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B = MazeCell.EMPTY, MazeCell.BLOCK
        R, M = MazeCell.ROBOT, MazeCell.OBJECT_BALL
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, B, B, E, B],
            [B, E, E, E, E, E, B],
            [B, R, M, E, B, B, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "SimpleRoom": [DistRewardSimpleRoom, GoalRewardSimpleRoom],
        "UMaze": [DistRewardUMaze, GoalRewardUMaze],
        "Push": [DistRewardPush, GoalRewardPush],
        "Fall": [DistRewardFall, GoalRewardFall],
        "2Rooms": [DistReward2Rooms, GoalReward2Rooms, SubGoal2Rooms],
        "4Rooms": [DistReward4Rooms, GoalReward4Rooms, SubGoal4Rooms],
        "TRoom": [DistRewardTRoom, GoalRewardTRoom, SubGoalTRoom],
        "BlockMaze": [DistRewardBlockMaze, GoalRewardBlockMaze],
        "Billiard": [
            DistRewardBilliard,
            GoalRewardBilliard,
            SubGoalBilliard,
            BanditBilliard,
        ],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
