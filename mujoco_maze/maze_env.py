# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Adapted from rllab maze_env.py."""

import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Tuple, Type

import gym
import numpy as np

from mujoco_maze import maze_env_utils, maze_task
from mujoco_maze.agent_model import AgentModel

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"


class MazeEnv(gym.Env):
    MODEL_CLASS: Type[AgentModel] = AgentModel
    MANUAL_COLLISION: bool = False

    def __init__(
        self,
        maze_task: Type[maze_task.MazeTask] = maze_task.SingleGoalSparseUMaze,
        n_bins: int = 0,
        sensor_range: float = 3.0,
        sensor_span: float = 2 * np.pi,
        observe_blocks: float = False,
        put_spin_near_agent: float = False,
        top_down_view: float = False,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        *args,
        **kwargs,
    ) -> None:
        self._task = maze_task(maze_size_scaling)

        xml_path = os.path.join(MODEL_DIR, self.MODEL_CLASS.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self._maze_height = height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self.t = 0  # time steps
        self._n_bins = n_bins
        self._sensor_range = sensor_range * size_scaling
        self._sensor_span = sensor_span
        self._observe_blocks = observe_blocks
        self._put_spin_near_agent = put_spin_near_agent
        self._top_down_view = top_down_view
        self._collision_coef = 0.1

        self._maze_structure = structure = self._task.create_maze()
        # Elevate the maze to allow for falling.
        self.elevated = any(maze_env_utils.MazeCell.CHASM in row for row in structure)
        # Are there any movable blocks?
        self.blocks = any(any(r.can_move() for r in row) for row in structure)

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self._init_positions = [
            (x - torso_x, y - torso_y) for x, y in self._find_all_robots()
        ]

        self._collision = maze_env_utils.Collision(
            structure, size_scaling, torso_x, torso_y,
        )

        self._xy_to_rowcol = lambda x, y: (
            2 + (y + size_scaling / 2) / size_scaling,
            2 + (x + size_scaling / 2) / size_scaling,
        )
        # walls (immovable), chasms (fall), movable blocks
        self._view = np.zeros([5, 5, 3])

        height_offset = 0.0
        if self.elevated:
            # Increase initial z-pos of ant.
            height_offset = height * size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set("pos", f"0 0 {0.75 + height_offset:.2f}")
        if self.blocks:
            # If there are movable blocks, change simulation settings to perform
            # better contact detection.
            default = tree.find(".//default")
            default.find(".//geom").set("solimp", ".995 .995 .01")

        self.movable_blocks = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]
                if struct.is_robot() and self._put_spin_near_agent:
                    struct = maze_env_utils.MazeCell.SpinXY
                if self.elevated and not struct.is_chasm():
                    # Create elevated platform.
                    x = j * size_scaling - torso_x
                    y = i * size_scaling - torso_y
                    h = height / 2 * size_scaling
                    size = 0.5 * size_scaling
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"elevated_{i}_{j}",
                        pos=f"{x} {y} {h}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1",
                    )
                if struct.is_block():
                    # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    x = j * size_scaling - torso_x
                    y = i * size_scaling - torso_y
                    h = height / 2 * size_scaling
                    size = 0.5 * size_scaling
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {h + height_offset}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                elif struct.can_move():
                    # Movable block.
                    # The "falling" blocks are shrunk slightly and increased in mass to
                    # ensure it can fall easily through a gap in the platform blocks.
                    name = "movable_%d_%d" % (i, j)
                    self.movable_blocks.append((name, struct))
                    falling = struct.can_move_z()
                    spinning = struct.can_spin()
                    shrink = 0.1 if spinning else 0.99 if falling else 1.0
                    height_shrink = 0.1 if spinning else 1.0
                    x = (
                        j * size_scaling - torso_x + 0.25 * size_scaling
                        if spinning
                        else 0.0
                    )
                    y = i * size_scaling - torso_y
                    h = height / 2 * size_scaling * height_shrink
                    size = 0.5 * size_scaling * shrink
                    movable_body = ET.SubElement(
                        worldbody,
                        "body",
                        name=name,
                        pos=f"{x} {y} {height_offset + h}",
                    )
                    ET.SubElement(
                        movable_body,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos="0 0 0",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        mass="0.001" if falling else "0.0002",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.1 0.1 1",
                    )
                    if struct.can_move_x():
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="1 0 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range=f"{-size_scaling} {size_scaling}",
                            margin="0.01",
                            name=f"movable_x_{i}_{j}",
                            pos="0 0 0",
                            type="slide",
                        )
                    if struct.can_move_y():
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="0 1 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range=f"{-size_scaling} {size_scaling}",
                            margin="0.01",
                            name=f"movable_y_{i}_{j}",
                            pos="0 0 0",
                            type="slide",
                        )
                    if struct.can_move_z():
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="true",
                            range=f"{-height_offset} 0",
                            margin="0.01",
                            name=f"movable_z_{i}_{j}",
                            pos="0 0 0",
                            type="slide",
                        )
                    if struct.can_spin():
                        ET.SubElement(
                            movable_body,
                            "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="false",
                            name=f"spinable_{i}_{j}",
                            pos="0 0 0",
                            type="ball",
                        )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if "name" not in geom.attrib:
                raise Exception("Every geom of the torso must have a name " "defined")

        # Set goals
        asset = tree.find(".//asset")
        for i, goal in enumerate(self._task.goals):
            ET.SubElement(asset, "material", name=f"goal{i}", rgba=goal.rbga_str())
            z = goal.pos[2] if goal.dim >= 3 else 0.0
            ET.SubElement(
                worldbody,
                "site",
                name=f"goal_site{i}",
                pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                size=f"{maze_size_scaling * 0.1}",
                material=f"goal{i}",
            )

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)
        self.world_tree = tree
        self.wrapped_env = self.MODEL_CLASS(*args, file_path=file_path, **kwargs)
        self.observation_space = self._get_obs_space()

    def get_ori(self) -> float:
        return self.wrapped_env.get_ori()

    def _get_obs_space(self) -> gym.spaces.Box:
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        # Set velocity limits
        wrapped_obs_space = self.wrapped_env.observation_space
        high[: wrapped_obs_space.shape[0]] = wrapped_obs_space.high
        low[: wrapped_obs_space.shape[0]] = wrapped_obs_space.low
        # Set coordinate limits
        low[0], high[0], low[1], high[1] = self._xy_limits()
        # Set orientation limits
        return gym.spaces.Box(low, high)

    def _xy_limits(self) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = 100, 100, -100, -100
        structure = self._maze_structure
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_block():
                continue
            xmin, xmax = min(xmin, j), max(xmax, j)
            ymin, ymax = min(ymin, i), max(ymax, i)
        x0, y0 = self._init_torso_x, self._init_torso_y
        scaling = self._maze_size_scaling
        xmin, xmax = (xmin - 0.5) * scaling - x0, (xmax + 0.5) * scaling - x0
        ymin, ymax = (ymin - 0.5) * scaling - y0, (ymax + 0.5) * scaling - y0
        return xmin, xmax, ymin, ymax

    def get_top_down_view(self):
        self._view = np.zeros_like(self._view)

        def valid(row, col):
            return self._view.shape[0] > row >= 0 and self._view.shape[1] > col >= 0

        def update_view(x, y, d, row=None, col=None):
            if row is None or col is None:
                x = x - self._robot_x
                y = y - self._robot_y

                row, col = self._xy_to_rowcol(x, y)
                update_view(x, y, d, row=row, col=col)
                return

            row, row_frac, col, col_frac = int(row), row % 1, int(col), col % 1
            if row_frac < 0:
                row_frac += 1
            if col_frac < 0:
                col_frac += 1

            if valid(row, col):
                self._view[row, col, d] += (
                    min(1.0, row_frac + 0.5) - max(0.0, row_frac - 0.5)
                ) * (min(1.0, col_frac + 0.5) - max(0.0, col_frac - 0.5))
            if valid(row - 1, col):
                self._view[row - 1, col, d] += (max(0.0, 0.5 - row_frac)) * (
                    min(1.0, col_frac + 0.5) - max(0.0, col_frac - 0.5)
                )
            if valid(row + 1, col):
                self._view[row + 1, col, d] += (max(0.0, row_frac - 0.5)) * (
                    min(1.0, col_frac + 0.5) - max(0.0, col_frac - 0.5)
                )
            if valid(row, col - 1):
                self._view[row, col - 1, d] += (
                    min(1.0, row_frac + 0.5) - max(0.0, row_frac - 0.5)
                ) * (max(0.0, 0.5 - col_frac))
            if valid(row, col + 1):
                self._view[row, col + 1, d] += (
                    min(1.0, row_frac + 0.5) - max(0.0, row_frac - 0.5)
                ) * (max(0.0, col_frac - 0.5))
            if valid(row - 1, col - 1):
                self._view[row - 1, col - 1, d] += (max(0.0, 0.5 - row_frac)) * max(
                    0.0, 0.5 - col_frac
                )
            if valid(row - 1, col + 1):
                self._view[row - 1, col + 1, d] += (max(0.0, 0.5 - row_frac)) * max(
                    0.0, col_frac - 0.5
                )
            if valid(row + 1, col + 1):
                self._view[row + 1, col + 1, d] += (max(0.0, row_frac - 0.5)) * max(
                    0.0, col_frac - 0.5
                )
            if valid(row + 1, col - 1):
                self._view[row + 1, col - 1, d] += (max(0.0, row_frac - 0.5)) * max(
                    0.0, 0.5 - col_frac
                )

        # Draw ant.
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        self._robot_x = robot_x
        self._robot_y = robot_y
        self._robot_ori = self.get_ori()

        structure = self._maze_structure
        size_scaling = self._maze_size_scaling

        # Draw immovable blocks and chasms.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j].is_block():  # Wall.
                    update_view(
                        j * size_scaling - self._init_torso_x,
                        i * size_scaling - self._init_torso_y,
                        0,
                    )
                if structure[i][j].is_chasm():  # Chasm.
                    update_view(
                        j * size_scaling - self._init_torso_x,
                        i * size_scaling - self._init_torso_y,
                        1,
                    )

        # Draw movable blocks.
        for block_name, block_type in self.movable_blocks:
            block_x, block_y = self.wrapped_env.get_body_com(block_name)[:2]
            update_view(block_x, block_y, 2)

        return self._view

    def get_range_sensor_obs(self):
        """Returns egocentric range sensor observations of maze."""
        robot_x, robot_y, robot_z = self.wrapped_env.get_body_com("torso")[:3]
        ori = self.get_ori()

        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        height = self._maze_height

        segments = []
        # Get line segments (corresponding to outer boundary) of each immovable
        # block or drop-off.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j].is_wall_or_chasm():  # There's a wall or drop-off.
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(segment=seg, type=structure[i][j],))
        # Get line segments (corresponding to outer boundary) of each movable
        # block within the agent's z-view.
        for block_name, block_type in self.movable_blocks:
            block_x, block_y, block_z = self.wrapped_env.get_body_com(block_name)[:3]
            if (
                block_z + height * size_scaling / 2 >= robot_z
                and robot_z >= block_z - height * size_scaling / 2
            ):  # Block in view.
                x1 = block_x - 0.5 * size_scaling
                x2 = block_x + 0.5 * size_scaling
                y1 = block_y - 0.5 * size_scaling
                y2 = block_y + 0.5 * size_scaling
                struct_segments = [
                    ((x1, y1), (x2, y1)),
                    ((x2, y1), (x2, y2)),
                    ((x2, y2), (x1, y2)),
                    ((x1, y2), (x1, y1)),
                ]
                for seg in struct_segments:
                    segments.append(dict(segment=seg, type=block_type))

        sensor_readings = np.zeros((self._n_bins, 3))  # 3 for wall, drop-off, block
        for ray_idx in range(self._n_bins):
            ray_ori = (
                ori
                - self._sensor_span * 0.5
                + (2 * ray_idx + 1.0) / (2 * self._n_bins) * self._sensor_span
            )
            ray_segments = []
            # Get all segments that intersect with ray.
            for seg in segments:
                p = maze_env_utils.ray_segment_intersect(
                    ray=((robot_x, robot_y), ray_ori), segment=seg["segment"]
                )
                if p is not None:
                    ray_segments.append(
                        dict(
                            segment=seg["segment"],
                            type=seg["type"],
                            ray_ori=ray_ori,
                            distance=maze_env_utils.point_distance(
                                p, (robot_x, robot_y)
                            ),
                        )
                    )
            if len(ray_segments) > 0:
                # Find out which segment is intersected first.
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                seg_type = first_seg["type"]
                idx = None
                if seg_type == 1:
                    idx = 0  # Wall
                elif seg_type == -1:
                    idx = 1  # Drop-off
                elif seg_type.can_move():
                    idx == 2  # Block
                sr = self._sensor_range
                if first_seg["distance"] <= sr:
                    sensor_readings[ray_idx][idx] = (sr - first_seg["distance"]) / sr

        return sensor_readings

    def _get_obs(self):
        wrapped_obs = self.wrapped_env._get_obs()
        if self._top_down_view:
            view = [self.get_top_down_view().flat]
        else:
            view = []

        if self._observe_blocks:
            additional_obs = []
            for block_name, block_type in self.movable_blocks:
                additional_obs.append(self.wrapped_env.get_body_com(block_name))
            wrapped_obs = np.concatenate(
                [wrapped_obs[:3]] + additional_obs + [wrapped_obs[3:]]
            )

        range_sensor_obs = self.get_range_sensor_obs()
        return np.concatenate(
            [wrapped_obs, range_sensor_obs.flat] + view + [[self.t * 0.001]]
        )

    def reset(self):
        self.t = 0
        self.wrapped_env.reset()
        # Sample a new goal
        if self._task.sample_goals():
            self.set_marker()
        if len(self._init_positions) > 1:
            xy = np.random.choice(self._init_positions)
            self.wrapped_env.set_xy(xy)
        return self._get_obs()

    def set_marker(self):
        for i, goal in enumerate(self._task.goals):
            idx = self.model.site_name2id(f"goal{i}")
            self.data.site_xpos[idx][: len(goal.pos)] = goal.pos

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def _find_robot(self):
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def _find_all_robots(self):
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        coords = []
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                coords.append((j * size_scaling, i * size_scaling))
        return coords

    def step(self, action):
        self.t += 1
        if self.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._collision.is_in(old_pos, new_pos):
                self.wrapped_env.set_xy(old_pos)
        else:
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        inner_reward = self._task.scale_inner_reward(inner_reward)
        outer_reward = self._task.reward(next_obs)
        done = self._task.termination(next_obs)
        return next_obs, inner_reward + outer_reward, done, info
