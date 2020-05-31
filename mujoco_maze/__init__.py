import gym

MAZE_IDS = ["Maze", "Push", "Fall"]  # TODO: Block, BlockMaze


def _get_kwargs(maze_id: str) -> tuple:
    return {
        "maze_id": maze_id,
        "observe_blocks": maze_id in ["Block", "BlockMaze"],
        "put_spin_near_agent": maze_id in ["Block", "BlockMaze"],
    }


for maze_id in MAZE_IDS:
    gym.envs.register(
        id="Ant{}-v0".format(maze_id),
        entry_point="mujoco_maze.ant_maze_env:AntMazeEnv",
        kwargs=dict(maze_size_scaling=8.0, **_get_kwargs(maze_id)),
        max_episode_steps=1000,
        reward_threshold=-1000,
    )
    gym.envs.register(
        id="Ant{}-v1".format(maze_id),
        entry_point="mujoco_maze.ant_maze_env:AntMazeEnv",
        kwargs=dict(maze_size_scaling=8.0, **_get_kwargs(maze_id)),
        max_episode_steps=1000,
        reward_threshold=0.9,
    )

for maze_id in MAZE_IDS:
    gym.envs.register(
        id="Point{}-v0".format(maze_id),
        entry_point="mujoco_maze.point_maze_env:PointMazeEnv",
        kwargs=_get_kwargs(maze_id),
        max_episode_steps=1000,
        reward_threshold=-1000,
    )
    gym.envs.register(
        id="Point{}-v1".format(maze_id),
        entry_point="mujoco_maze.point_maze_env:PointMazeEnv",
        kwargs=dict(**_get_kwargs(maze_id), dense_reward=False),
        max_episode_steps=1000,
        reward_threshold=0.9,
    )


__version__ = "0.1.0"
