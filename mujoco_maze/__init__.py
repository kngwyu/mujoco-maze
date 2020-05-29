import gym

MAZE_IDS = ["Maze", "Push", "Fall"]  # TODO: Block, BlockMaze


def _get_kwargs(maze_id: str) -> tuple:
    return {
        "observe_blocks": maze_id in ["Block", "BlockMaze"],
        "put_spin_near_agent": maze_id in ["Block", "BlockMaze"],
    }


for maze_id in MAZE_IDS:
    gym.envs.register(
        id="AntMaze{}-v0".format(maze_id),
        entry_point="mujoco_maze.ant_maze_env:AntMazeEnv",
        kwargs=dict(maze_id=maze_id, maze_size_scaling=8, **_get_kwargs(maze_id)),
        max_episode_steps=1000,
        reward_threshold=-1000,
    )

for maze_id in MAZE_IDS:
    gym.envs.register(
        id="PointMaze{}-v0".format(maze_id),
        entry_point="mujoco_maze.point_maze_env:PointMazeEnv",
        kwargs=dict(
            maze_id=maze_id,
            maze_size_scaling=4,
            manual_collision=True,
            **_get_kwargs(maze_id),
        ),
        max_episode_steps=1000,
        reward_threshold=-1000,
    )


__version__ = "0.1.0"
