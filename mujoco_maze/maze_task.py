from abc import ABC, abstractmethod
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell


class MazeGoal:
    THRESHOLD: float = 0.6

    def __init__(self, goal: np.ndarray, reward_scale: float = 1.0) -> None:
        self.goal = goal
        self.goal_dim = goal.shape[0]
        self.reward_scale = reward_scale

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.goal_dim] - self.goal) <= self.THRESHOLD

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.goal_dim] - self.goal)) ** 0.5


class MazeTask(ABC):
    REWARD_THRESHOLD: float

    def __init__(self) -> None:
        self.goals = []

    @abstractmethod
    def sample_goals(self, scale: float) -> None:
        pass

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @abstractmethod
    def termination(self, obs: np.ndarray) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class SingleGoalSparseEMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9

    def sample_goals(self, scale: float) -> None:
        goal = MazeGoal(np.array([0.0, 2.0 * scale]))
        self.goals = [goal]

    def reward(self, obs: np.ndarray) -> float:
        if self.goals[0].neighbor(obs):
            return 1.0
        else:
            return -0.0001

    def termination(self, obs: np.ndarray) -> bool:
        return self.goals[0].neighbor(obs)

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


class SingleGoalDenseEMaze(SingleGoalSparseEMaze):
    REWARD_THRESHOLD: float = 1000.0

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs)


class SingleGoalSparsePush(SingleGoalSparseEMaze):
    def sample_goals(self, scale: float) -> None:
        goal = MazeGoal(np.array([0.0, 2.375 * scale]))
        self.goals = [goal]

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


class SingleGoalSparseFall(SingleGoalSparseEMaze):
    def sample_goals(self, scale: float) -> None:
        goal = MazeGoal(np.array([0.0, 3.375 * scale, 4.5]))
        self.goals = [goal]

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


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Maze": [SingleGoalDenseEMaze, SingleGoalSparseEMaze],
        "Push": [SingleGoalDensePush, SingleGoalSparsePush],
        "Fall": [SingleGoalDenseFall, SingleGoalSparseFall],
    }
