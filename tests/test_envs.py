import gym
import pytest

import mujoco_maze


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_ant_maze(maze_id):
    if "Billiard" in maze_id:
        return
    for i in range(2):
        env = gym.make(f"Ant{maze_id}-v{i}")
        s0 = env.reset()
        s, _, _, _ = env.step(env.action_space.sample())
        if not env.unwrapped.has_extended_obs:
            assert s0.shape == (30,)
            assert s.shape == (30,)


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_point_maze(maze_id):
    for i in range(2):
        env = gym.make(f"Point{maze_id}-v{i}")
        s0 = env.reset()
        s, r, _, _ = env.step(env.action_space.sample())
        if not env.unwrapped.has_extended_obs:
            assert s0.shape == (7,)
            assert s.shape == (7,)
        if env.unwrapped._observe_balls:
            assert s0.shape == (10,)
            assert s.shape == (10,)
        if i == 0:
            assert r != 0.0
        else:
            assert r == env.unwrapped._task.PENALTY
            assert r < 0.0


@pytest.mark.parametrize("maze_id", ["2Rooms", "4Rooms", "Billiard"])
def test_subgoal_envs(maze_id):
    env = gym.make(f"Point{maze_id}-v2")
    s0 = env.reset()
    s, r, _, _ = env.step(env.action_space.sample())
    if not env.unwrapped.has_extended_obs:
        assert s0.shape == (7,)
        assert s.shape == (7,)
    elif env.unwrapped._observe_balls:
        assert s0.shape == (10,)
        assert s.shape == (10,)
    assert len(env.unwrapped._task.goals) > 1


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_reacher_maze(maze_id):
    for inhibited in ["Fall", "Push", "Block", "Billiard"]:
        if inhibited in maze_id:
            return
    for i in range(2):
        env = gym.make(f"Reacher{maze_id}-v{i}")
        s0 = env.reset()
        s, _, _, _ = env.step(env.action_space.sample())
        if not env.unwrapped.has_extended_obs:
            assert s0.shape == (9,)
            assert s.shape == (9,)


@pytest.mark.parametrize("maze_id", mujoco_maze.TaskRegistry.keys())
def test_swimmer_maze(maze_id):
    for inhibited in ["Fall", "Push", "Block", "Billiard"]:
        if inhibited in maze_id:
            return
    for i in range(2):
        env = gym.make(f"Swimmer{maze_id}-v{i}")
        s0 = env.reset()
        s, _, _, _ = env.step(env.action_space.sample())
        if not env.unwrapped.has_extended_obs:
            assert s0.shape == (11,)
            assert s.shape == (11,)


@pytest.mark.parametrize("v", [0, 1])
def test_maze_args(v):
    env = gym.make(f"PointTRoom-v{v}", task_kwargs={"goal": (-2.0, -3.0)})
    assert env.reset().shape == (7,)
    s, _, _, _ = env.step(env.action_space.sample())
    assert s.shape == (7,)
