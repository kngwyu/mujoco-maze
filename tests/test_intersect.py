import numpy as np
import pytest

from mujoco_maze.maze_env_utils import Line


@pytest.mark.parametrize(
    "l1, l2, p, ans",
    [
        ((0.0, 0.0), (4.0, 4.0), (1.0, 3.0), 2.0 ** 0.5),
        ((-3.0, -3.0), (0.0, 1.0), (-3.0, 1.0), 2.4),
    ],
)
def test_distance(l1, l2, p, ans):
    line = Line(l1, l2)
    point = complex(*p)
    assert abs(line.distance(point) - ans) <= 1e-8


@pytest.mark.parametrize(
    "l1p1, l1p2, l2p1, l2p2, none",
    [
        ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), (1.0, 1.0), False),
        ((1.0, 1.0), (2.0, 3.0), (-1.0, 1.5), (1.5, 1.0), False),
        ((1.5, 1.5), (2.0, 3.0), (-1.0, 1.5), (1.5, 1.0), True),
        ((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 3.0), False),
    ],
)
def test_intersect(l1p1, l1p2, l2p1, l2p2, none):
    l1 = Line(l1p1, l1p2)
    l2 = Line(l2p1, l2p2)
    i1 = l1.intersect(l2)
    i2 = line_intersect(l1p1, l1p2, l2p1, l2p2)
    if none:
        assert i1 is None and i2 is None
    else:
        assert i1 is not None
        i1 = np.array([i1.real, i1.imag])
        np.testing.assert_array_almost_equal(i1, np.array(i2))


def line_intersect(pt1, pt2, ptA, ptB):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html
    Returns the intersection of Line(pt1,pt2) and Line(ptA,ptB).
    """
    import math

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
        return None

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    if r >= 0 and 0 <= s <= 1:
        return xi, yi
    else:
        return None
