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
Point = complex


class MazeCell(Enum):
    # Robot: Start position
    ROBOT = -1
    # Blocks
    EMPTY = 0
    BLOCK = 1
    CHASM = 2
    OBJECT_BALL = 3
    # Moves
    XY_BLOCK = 14
    XZ_BLOCK = 15
    YZ_BLOCK = 16
    XYZ_BLOCK = 17
    XY_HALF_BLOCK = 18
    SPIN = 19

    def is_block(self) -> bool:
        return self == self.BLOCK

    def is_chasm(self) -> bool:
        return self == self.CHASM

    def is_object_ball(self) -> bool:
        return self == self.OBJECT_BALL

    def is_empty(self) -> bool:
        return self == self.ROBOT or self == self.EMPTY

    def is_robot(self) -> bool:
        return self == self.ROBOT

    def is_wall_or_chasm(self) -> bool:
        return self in [self.BLOCK, self.CHASM]

    def can_move_x(self) -> bool:
        return self in [
            self.XY_BLOCK,
            self.XY_HALF_BLOCK,
            self.XZ_BLOCK,
            self.XYZ_BLOCK,
            self.SPIN,
        ]

    def can_move_y(self) -> bool:
        return self in [
            self.XY_BLOCK,
            self.XY_HALF_BLOCK,
            self.YZ_BLOCK,
            self.XYZ_BLOCK,
            self.SPIN,
        ]

    def can_move_z(self) -> bool:
        return self in [self.XZ_BLOCK, self.YZ_BLOCK, self.XYZ_BLOCK]

    def can_spin(self) -> bool:
        return self == self.SPIN

    def can_move(self) -> bool:
        return self.can_move_x() or self.can_move_y() or self.can_move_z()

    def is_half_block(self) -> bool:
        return self in [self.XY_HALF_BLOCK]


class Line:
    def __init__(
        self,
        p1: Union[Sequence[float], Point],
        p2: Union[Sequence[float], Point],
    ) -> None:
        self.p1 = p1 if isinstance(p1, Point) else complex(*p1)
        self.p2 = p2 if isinstance(p2, Point) else complex(*p2)
        self.v1 = self.p2 - self.p1
        self.conj_v1 = self.v1.conjugate()
        self.norm = abs(self.v1)

    def _intersect(self, other: Self) -> bool:
        v2 = other.p1 - self.p1
        v3 = other.p2 - self.p1
        return (self.conj_v1 * v2).imag * (self.conj_v1 * v3).imag <= 0.0

    def _projection(self, p: Point) -> Point:
        nv1 = -self.v1
        nv1_norm = abs(nv1) ** 2
        scale = ((p - self.p1).conjugate() * nv1).real / nv1_norm
        return self.p1 + nv1 * scale

    def reflection(self, p: Point) -> Point:
        return p + 2.0 * (self._projection(p) - p)

    def distance(self, p: Point) -> float:
        return abs(p - self._projection(p))

    def intersect(self, other: Self) -> Point:
        if self._intersect(other) and other._intersect(self):
            return self._cross_point(other)
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
    def __init__(self, point: Point, reflection: Point) -> None:
        self._point = point
        self._reflection = reflection

    @property
    def point(self) -> np.ndarray:
        return np.array([self._point.real, self._point.imag])

    def rest(self) -> np.ndarray:
        p = self._reflection - self._point
        return np.array([p.real, p.imag])


class CollisionDetector:
    """For manual collision detection."""

    EPS: float = 0.05
    NEIGHBORS: List[Tuple[int, int]] = [[0, -1], [-1, 0], [0, 1], [1, 0]]

    def __init__(
        self,
        structure: list,
        size_scaling: float,
        torso_x: float,
        torso_y: float,
        radius: float,
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
            offset = size_scaling * 0.5 + radius
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

    def detect(self, old_pos: np.ndarray, new_pos: np.ndarray) -> Optional[Collision]:
        move = Line(old_pos, new_pos)
        # First, checks that it actually moved
        if move.norm <= 1e-8:
            return None
        # Next, checks that the trajectory cross the wall or not
        collisions = []
        for line in self.lines:
            intersection = line.intersect(move)
            if intersection is not None:
                reflection = line.reflection(move.p2)
                collisions.append(Collision(intersection, reflection))
        if len(collisions) == 0:
            return None
        col = collisions[0]
        dist = abs(col._point - move.p1)
        for collision in collisions[1:]:
            new_dist = abs(collision._point - move.p1)
            if new_dist < dist:
                col, dist = collision, new_dist
        return col
