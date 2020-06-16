import gym

from mujoco_maze.maze_task import TaskRegistry


MAZE_IDS = ["Maze", "Push", "Fall"]  # TODO: Block, BlockMaze


def _get_kwargs(maze_id: str) -> tuple:
    return {
        "maze_id": maze_id,
        "observe_blocks": maze_id in ["Block", "BlockMaze"],
        "put_spin_near_agent": maze_id in ["Block", "BlockMaze"],
    }


for maze_id in MAZE_IDS:
    for i, task_cls in enumerate(TaskRegistry.REGISTRY[maze_id]):
        gym.envs.register(
            id=f"Ant{maze_id}-v{i}",
            entry_point="mujoco_maze.ant_maze_env:AntMazeEnv",
            kwargs=dict(maze_task=task_cls, maze_size_scaling=8.0),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )

for maze_id in MAZE_IDS:
    for i, task_cls in enumerate(TaskRegistry.REGISTRY[maze_id]):
        gym.envs.register(
            id=f"Point{maze_id}-v{i}",
            entry_point="mujoco_maze.point_maze_env:PointMazeEnv",
            kwargs=dict(maze_task=task_cls),
            max_episode_steps=1000,
            reward_threshold=task_cls.REWARD_THRESHOLD,
        )


__version__ = "0.1.0"
