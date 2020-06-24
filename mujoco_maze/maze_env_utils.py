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

"""Adapted from rllab maze_env_utils.py."""
import itertools as it
import math
from enum import Enum

import numpy as np


class MazeCell(Enum):
    # Robot: Start position
    ROBOT = -1
    # Blocks
    EMPTY = 0
    BLOCK = 1
    CHASM = 2
    # Moves
    X = 11
    Y = 12
    Z = 13
    XY = 14
    XZ = 15
    YZ = 16
    XYZ = 17
    SpinXY = 18

    def is_block(self) -> bool:
        return self == self.BLOCK

    def is_chasm(self) -> bool:
        return self == self.CHASM

    def is_robot(self) -> bool:
        return self == self.ROBOT

    def is_wall_or_chasm(self) -> bool:
        return self in [self.BLOCK, self.CHASM]

    def can_move_x(self) -> bool:
        return self in [
            self.X,
            self.XY,
            self.XZ,
            self.XYZ,
            self.SpinXY,
        ]

    def can_move_y(self):
        return self in [
            self.Y,
            self.XY,
            self.YZ,
            self.XYZ,
            self.SpinXY,
        ]

    def can_move_z(self):
        return self in [self.Z, self.XZ, self.YZ, self.XYZ]

    def can_spin(self):
        return self == self.SpinXY

    def can_move(self):
        return self.can_move_x() or self.can_move_y() or self.can_move_z()


class Collision:
    """For manual collision detection.
    """

    ARROUND = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    OFFSET = {False: 0.48, True: 0.51}

    def __init__(
        self, structure: list, size_scaling: float, torso_x: float, torso_y: float,
    ) -> None:
        h, w = len(structure), len(structure[0])
        self.objects = []

        def is_block(pos) -> bool:
            i, j = pos
            if 0 <= i < h and 0 <= j < w:
                return structure[i][j].is_block()
            else:
                return False

        def offset(pos, index) -> float:
            return self.OFFSET[is_block(pos + self.ARROUND[index])]

        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if not structure[i][j].is_block():
                continue
            pos = np.array([i, j])
            y_base = i * size_scaling - torso_y
            min_y = y_base - size_scaling * offset(pos, 0)
            max_y = y_base + size_scaling * offset(pos, 1)
            x_base = j * size_scaling - torso_x
            min_x = x_base - size_scaling * offset(pos, 2)
            max_x = x_base + size_scaling * offset(pos, 3)
            self.objects.append((min_y, max_y, min_x, max_x))

    def is_in(self, old_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        # Heuristics to prevent the agent from going through the wall
        for x, y in ((old_pos + new_pos) / 2, new_pos):
            for min_y, max_y, min_x, max_x in self.objects:
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    return True
        return False


def line_intersect(pt1, pt2, ptA, ptB):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html
    Returns the intersection of Line(pt1,pt2) and Line(ptA,ptB).
    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = -dx1 * dy + dy1 * dx

    if math.fabs(DET) < DET_TOLERANCE:
        return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def ray_segment_intersect(ray, segment):
    """
    Check if the ray originated from (x, y) with direction theta intersect the line
    segment (x1, y1) -- (x2, y2), and return the intersection point if there is one.
    """
    (x, y), theta = ray
    # (x1, y1), (x2, y2) = segment
    pt1 = (x, y)
    pt2 = (x + math.cos(theta), y + math.sin(theta))
    xo, yo, valid, r, s = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and 0 <= s <= 1:
        return (xo, yo)
    return None


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
