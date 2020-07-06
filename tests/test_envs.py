import gym
import numpy as np
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


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_collision_lines(maze_id):
    env = gym.make(f"Point{maze_id}-v0")
    if maze_id == "UMaze":
        assert len(env.unwrapped._collision.lines) == 16
    structure = env.unwrapped._maze_structure
    scaling = env.unwrapped._maze_size_scaling
    init_x = env.unwrapped._init_torso_x
    init_y = env.unwrapped._init_torso_y

    def check_pos(pos):
        x_orig = (pos.real + init_x) / scaling
        y_orig = (pos.imag + init_y) / scaling
        return structure[int(round(y_orig))][int(round(x_orig))]

    for line in env.unwrapped._collision.lines:
        mid = (line.p1 + line.p2) / 2
        p2p1 = line.p2 - line.p1
        cell1 = check_pos(mid + 0.1 * p2p1 * np.complex(0.0, -1.0))
        cell2 = check_pos(mid + 0.1 * p2p1 * np.complex(0.0, 1.0))
        if cell1.is_block():
            assert not cell2.is_block()
        else:
            assert cell2.is_block()
