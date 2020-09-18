import gym
import pytest

import mujoco_maze


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_ant_maze(maze_id):
    for i in range(2):
        env = gym.make(f"Ant{maze_id}-v{i}")
        assert env.reset().shape == (30,)
        s, _, _, _ = env.step(env.action_space.sample())
        assert s.shape == (30,)


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_point_maze(maze_id):
    for i in range(2):
        env = gym.make(f"Point{maze_id}-v{i}")
        assert env.reset().shape == (7,)
        s, _, _, _ = env.step(env.action_space.sample())
        assert s.shape == (7,)


@pytest.mark.parametrize("v", [0, 1])
def test_maze_args(v):
    env = gym.make(f"PointTRoom-v{v}", task_kwargs={"goals": [(-2.0, -4.0)]})
    assert env.reset().shape == (7,)
    s, _, _, _ = env.step(env.action_space.sample())
    assert s.shape == (7,)
