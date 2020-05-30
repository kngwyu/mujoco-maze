import gym
import mujoco_maze
import pytest


@pytest.mark.parametrize("maze_id", mujoco_maze.MAZE_IDS)
def test_ant_maze(maze_id):
    env = gym.make("Ant{}-v0".format(maze_id))
    assert env.reset().shape == (30,)


@pytest.mark.parametrize("maze_id", mujoco_maze.MAZE_IDS)
def test_point_maze(maze_id):
    env = gym.make("Point{}-v0".format(maze_id))
    assert env.reset().shape == (7,)
