"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""


import gym

from mujoco_maze.ant import AntEnv
from mujoco_maze.maze_task import TaskRegistry
from mujoco_maze.point import PointEnv
from mujoco_maze.reacher import ReacherEnv
from mujoco_maze.swimmer import SwimmerEnv

for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        # Point
        gym.envs.register(
            id=f"Point{maze_id}-v{i}",
            entry_point="mujoco_maze.maze_env:MazeEnv",
            kwargs=dict(
                model_cls=PointEnv,
                maze_task=task_cls,
                maze_size_scaling=task_cls.MAZE_SIZE_SCALING.point,
                inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
            ),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )
        if "Billiard" in maze_id:
            continue
        # Ant
        gym.envs.register(
            id=f"Ant{maze_id}-v{i}",
            entry_point="mujoco_maze.maze_env:MazeEnv",
            kwargs=dict(
                model_cls=AntEnv,
                maze_task=task_cls,
                maze_size_scaling=task_cls.MAZE_SIZE_SCALING.ant,
                inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
            ),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )
        skip_swimmer = False
        for inhibited in ["Fall", "Push", "Block"]:
            if inhibited in maze_id:
                skip_swimmer = True

        if skip_swimmer:
            continue

        # Reacher
        gym.envs.register(
            id=f"Reacher{maze_id}-v{i}",
            entry_point="mujoco_maze.maze_env:MazeEnv",
            kwargs=dict(
                model_cls=ReacherEnv,
                maze_task=task_cls,
                maze_size_scaling=task_cls.MAZE_SIZE_SCALING.swimmer,
                inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
            ),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )

        # Swimmer
        gym.envs.register(
            id=f"Swimmer{maze_id}-v{i}",
            entry_point="mujoco_maze.maze_env:MazeEnv",
            kwargs=dict(
                model_cls=SwimmerEnv,
                maze_task=task_cls,
                maze_size_scaling=task_cls.MAZE_SIZE_SCALING.swimmer,
                inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
            ),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )


__version__ = "0.1.0"
