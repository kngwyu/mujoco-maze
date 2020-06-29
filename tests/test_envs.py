import gym
import pytest

import mujoco_maze


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_ant_maze(maze_id):
    env = gym.make("Ant{}-v0".format(maze_id))
    assert env.reset().shape == (30,)
    s, _, _, _ = env.step(env.action_space.sample())
    assert s.shape == (30,)


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_point_maze(maze_id):
    env = gym.make("Point{}-v0".format(maze_id))
    assert env.reset().shape == (7,)
    s, _, _, _ = env.step(env.action_space.sample())
    assert s.shape == (7,)
