"""
Utilities for creating maze.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

import itertools as it
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

Self = Any
Point = np.complex


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

    def is_empty(self) -> bool:
        return self == self.ROBOT or self == self.EMPTY

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


class Line:
    def __init__(
        self, p1: Union[Point, Sequence[float]], p2: Union[Point, Sequence[float]]
    ) -> None:
        if isinstance(p1, Point):
            self.p1 = p1
        else:
            self.p1 = np.complex(*p1)
        if isinstance(p2, Point):
            self.p2 = p2
        else:
            self.p2 = np.complex(*p2)
        self.conj_v1 = np.conjugate(self.p2 - self.p1)

    def extend(self, dist: float) -> Tuple[Self, Point]:
        v = self.p2 - self.p1
        extended_v = v * dist / np.absolute(v)
        p2 = self.p2 + extended_v
        return Line(self.p1, p2), extended_v

    def _intersect(self, other: Self) -> bool:
        v2 = other.p1 - self.p1
        v3 = other.p2 - self.p1
        return (self.conj_v1 * v2).imag * (self.conj_v1 * v3).imag <= 0.0

    def intersect(self, other: Self) -> Optional[np.ndarray]:
        if self._intersect(other) and other._intersect(self):
            cross = self._cross_point(other)
            return np.array([cross.real, cross.imag])
        else:
            return None

    def _cross_point(self, other: Self) -> Optional[Point]:
        v2 = other.p2 - other.p1
        v3 = self.p2 - other.p1
        a, b = (self.conj_v1 * v2).imag, (self.conj_v1 * v3).imag
        return other.p1 + b / a * v2

    def __repr__(self) -> str:
        x1, y1 = self.p1.real, self.p1.imag
        x2, y2 = self.p2.real, self.p2.imag
        return f"Line(({x1}, {y1}) -> ({x2}, {y2}))"


class Collision:
    """For manual collision detection.
    """

    NEIGHBORS: List[Tuple[int, int]] = [[0, -1], [-1, 0], [0, 1], [1, 0]]

    def __init__(
        self, structure: list, size_scaling: float, torso_x: float, torso_y: float,
    ) -> None:
        h, w = len(structure), len(structure[0])
        self.lines = []

        def is_empty(i, j) -> bool:
            if 0 <= i < h and 0 <= j < w:
                return structure[i][j].is_empty()
            else:
                return False

        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if not structure[i][j].is_block():
                continue
            y_base = i * size_scaling - torso_y
            x_base = j * size_scaling - torso_x
            offset = size_scaling * 0.5
            min_y, max_y = y_base - offset, y_base + offset
            min_x, max_x = x_base - offset, x_base + offset
            for dx, dy in self.NEIGHBORS:
                if not is_empty(i + dy, j + dx):
                    continue
                self.lines.append(
                    Line(
                        (max_x if dx == 1 else min_x, max_y if dy == 1 else min_y),
                        (min_x if dx == -1 else max_x, min_y if dy == -1 else max_y),
                    )
                )

    def detect_intersection(
        self, old_pos: np.ndarray, new_pos: np.ndarray, radius
    ) -> Optional[np.ndarray]:
        move, extended = Line(old_pos, new_pos).extend(radius)
        intersections = []
        for line in self.lines:
            intersection = line.intersect(move)
            if intersection is not None:
                intersections.append(intersection)
        if len(intersections) == 0:
            return None
        pos = intersections[0]
        dist = np.linalg.norm(pos - old_pos)
        for new_pos in intersections[1:]:
            new_dist = np.linalg.norm(new_pos - old_pos)
            if new_dist < dist:
                pos, dist = new_pos, new_dist
        return pos - np.array([extended.real, extended.imag])
