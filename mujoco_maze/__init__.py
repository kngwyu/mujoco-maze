import gym

from mujoco_maze.maze_task import TaskRegistry


for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        gym.envs.register(
            id=f"Ant{maze_id}-v{i}",
            entry_point="mujoco_maze.ant_maze_env:AntMazeEnv",
            kwargs=dict(
                maze_task=task_cls,
                maze_size_scaling=task_cls.MAZE_SIZE_SCALING.ant,
                inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
            ),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )

for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        gym.envs.register(
            id=f"Point{maze_id}-v{i}",
            entry_point="mujoco_maze.point_maze_env:PointMazeEnv",
            kwargs=dict(
                maze_task=task_cls,
                maze_size_scaling=task_cls.MAZE_SIZE_SCALING.point,
                inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
            ),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )


__version__ = "0.1.0"
