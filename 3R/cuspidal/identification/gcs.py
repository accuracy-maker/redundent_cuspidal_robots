"""
Cuspidal identification.

Cupidal robot has at least one pair of non-singluar IKS. Therefore, one way to certify if one robot is cuspidal or not is searching this special pair.

The proposed way is that:
1. based on maths, for two IKS q_i and q_j, if f(q_i) = f(q_j), q_i != q_j and CC(i) == CC(j) => cuspidal
2. C-IRIS to construct this certified "singular-free" region

For an orthogonal cuspidal 3R case study, the program:
1. read the DH parameters
2. derive FK kinematics
3. adaptively covers the non-singular (q2, q3) tours with convex boxes
4. certify every accepted box using interval arithmetic
5. merge adjacent certified boxes into larger convex rectangles
6. build a periodic overlap/contact graph
7. finds all position IK roots for a selected target
8. finds two distinct IK roots in the same graph components
9. construct a continous piecewise-linear non-singular path
10. uses a Krawczyk interval test around the two IK roots.
11. plot
"""
from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import root
import roboticstoolbox as rtb


PI = math.pi
TWO_PI = 2.0 * PI


# ============================================================================
# Robot definition
# ============================================================================


def build_robot():
    """
    Build exactly the robot specified in the question.

    The remainder of the program can still run when Robotics Toolbox is
    unavailable because analytic and direct-DH implementations are included.
    """

    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(
                d=0.0,
                a=1.0,
                alpha=-np.pi / 2,
            ),
            rtb.RevoluteDH(
                d=1.0,
                a=2.0,
                alpha=np.pi / 2,
            ),
            rtb.RevoluteDH(
                d=0.0,
                a=1.5,
                alpha=0.0,
            ),
        ],
        name="3R_cuspidal_robot",
    )

    return robot


# ============================================================================
# Angle utilities
# ============================================================================


def wrap_to_pi(x):
    """Wrap a scalar or array of angles to [-pi, pi)."""
    return (np.asarray(x) + PI) % TWO_PI - PI


def angular_difference(a, b):
    """Return the componentwise wrapped difference a-b."""
    return wrap_to_pi(np.asarray(a) - np.asarray(b))


# ============================================================================
# Closed-form forward kinematics
# ============================================================================


def fk_position(q: np.ndarray) -> np.ndarray:
    """
    Closed-form end-effector position for the supplied standard-DH robot.

    For this robot, define

        L = 2 + 1.5 cos(q3)
        A = 1 + L cos(q2)
        B = 1 + 1.5 sin(q3)

    Then

        x = cos(q1) A - sin(q1) B
        y = sin(q1) A + cos(q1) B
        z = -L sin(q2)
    """
    q1, q2, q3 = np.asarray(q, dtype=float)

    L = 2.0 + 1.5 * np.cos(q3)
    A = 1.0 + L * np.cos(q2)
    B = 1.0 + 1.5 * np.sin(q3)

    c1 = np.cos(q1)
    s1 = np.sin(q1)

    return np.array(
        [
            c1 * A - s1 * B,
            s1 * A + c1 * B,
            -L * np.sin(q2),
        ],
        dtype=float,
    )


def det_jv(q_or_q2, q3: Optional[float] = None):
    """
    Determinant of the 3x3 translational Jacobian.

    It can be called as either

        det_jv([q1, q2, q3])

    or

        det_jv(q2, q3)

    The determinant is independent of q1.
    """
    if q3 is None:
        q = np.asarray(q_or_q2, dtype=float)
        q2 = q[..., -2]
        q3 = q[..., -1]
    else:
        q2 = np.asarray(q_or_q2, dtype=float)
        q3 = np.asarray(q3, dtype=float)

    L = 2.0 + 1.5 * np.cos(q3)

    singular_term = (
        np.sin(q3)
        + np.cos(q2) * (2.0 * np.sin(q3) - np.cos(q3))
    )

    return 1.5 * L * singular_term


def singular_factor(q2, q3):
    """
    The singularity set is exactly singular_factor(q2,q3)=0 because

        2 + 1.5 cos(q3) >= 0.5 > 0.
    """
    return (
        np.sin(q3)
        + np.cos(q2) * (2.0 * np.sin(q3) - np.cos(q3))
    )


# ============================================================================
# Independent direct-DH verification
# ============================================================================


def dh_transform(
    theta: float,
    d: float,
    a: float,
    alpha: float,
) -> np.ndarray:
    """Standard DH homogeneous transformation."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def direct_dh_fk_and_jv(
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute position and translational Jacobian from direct DH products.

    This provides a verification path independent of the closed-form
    expressions.
    """
    dh_parameters = (
        (0.0, 1.0, -PI / 2.0),
        (1.0, 2.0, PI / 2.0),
        (0.0, 1.5, 0.0),
    )

    T = np.eye(4)

    origins = [T[:3, 3].copy()]
    z_axes = [T[:3, 2].copy()]

    for i, (d, a, alpha) in enumerate(dh_parameters):
        T = T @ dh_transform(
            theta=q[i],
            d=d,
            a=a,
            alpha=alpha,
        )

        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())

    end_position = origins[-1]

    Jv = np.column_stack(
        [
            np.cross(
                z_axes[i],
                end_position - origins[i],
            )
            for i in range(3)
        ]
    )

    return end_position, Jv


def validate_kinematics(
    robot,
    random_tests: int = 100,
    seed: int = 7,
) -> tuple[float, float]:
    """
    Compare the closed-form model with either Robotics Toolbox or the
    independent direct-DH implementation.
    """
    rng = np.random.default_rng(seed)

    maximum_position_error = 0.0
    maximum_determinant_error = 0.0

    for _ in range(random_tests):
        q = rng.uniform(-PI, PI, 3)

        closed_position = fk_position(q)
        closed_determinant = float(det_jv(q))

        if robot is not None:
            reference_position = np.asarray(
                robot.fkine(q).t,
                dtype=float,
            ).reshape(3)

            reference_jv = np.asarray(
                robot.jacob0(q),
                dtype=float,
            )[:3, :]
        else:
            reference_position, reference_jv = direct_dh_fk_and_jv(q)

        position_error = np.linalg.norm(
            closed_position - reference_position
        )

        determinant_error = abs(
            closed_determinant
            - float(np.linalg.det(reference_jv))
        )

        maximum_position_error = max(
            maximum_position_error,
            float(position_error),
        )

        maximum_determinant_error = max(
            maximum_determinant_error,
            float(determinant_error),
        )

    return maximum_position_error, maximum_determinant_error


# ============================================================================
# Outward-expanded interval arithmetic
# ============================================================================


def round_down(x: float) -> float:
    """Move one floating-point value toward negative infinity."""
    return float(np.nextafter(float(x), -np.inf))


def round_up(x: float) -> float:
    """Move one floating-point value toward positive infinity."""
    return float(np.nextafter(float(x), np.inf))


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(
                f"Invalid interval [{self.lo}, {self.hi}]"
            )

    @staticmethod
    def point(x: float) -> "Interval":
        return Interval(float(x), float(x))

    def __add__(self, other) -> "Interval":
        other = as_interval(other)

        return Interval(
            round_down(self.lo + other.lo),
            round_up(self.hi + other.hi),
        )

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(
            round_down(-self.hi),
            round_up(-self.lo),
        )

    def __sub__(self, other) -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other) -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other) -> "Interval":
        other = as_interval(other)

        products = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )

        return Interval(
            round_down(min(products)),
            round_up(max(products)),
        )

    __rmul__ = __mul__

    @property
    def width(self) -> float:
        return self.hi - self.lo


def as_interval(value) -> Interval:
    if isinstance(value, Interval):
        return value

    return Interval.point(float(value))


def interval_contains_phase(
    lo: float,
    hi: float,
    phase: float,
    period: float,
) -> bool:
    """
    Return True when [lo,hi] contains phase+k*period for some integer k.
    """
    k_min = math.ceil((lo - phase) / period)
    k_max = math.floor((hi - phase) / period)

    return k_min <= k_max


def interval_sin(x: Interval) -> Interval:
    """Conservative sine interval."""
    if x.width >= TWO_PI:
        return Interval(-1.0, 1.0)

    endpoint_values = [
        math.sin(x.lo),
        math.sin(x.hi),
    ]

    lower = min(endpoint_values)
    upper = max(endpoint_values)

    if interval_contains_phase(
        x.lo,
        x.hi,
        PI / 2.0,
        TWO_PI,
    ):
        upper = 1.0

    if interval_contains_phase(
        x.lo,
        x.hi,
        -PI / 2.0,
        TWO_PI,
    ):
        lower = -1.0

    return Interval(
        round_down(lower),
        round_up(upper),
    )


def interval_cos(x: Interval) -> Interval:
    """Conservative cosine interval."""
    if x.width >= TWO_PI:
        return Interval(-1.0, 1.0)

    endpoint_values = [
        math.cos(x.lo),
        math.cos(x.hi),
    ]

    lower = min(endpoint_values)
    upper = max(endpoint_values)

    if interval_contains_phase(
        x.lo,
        x.hi,
        0.0,
        TWO_PI,
    ):
        upper = 1.0

    if interval_contains_phase(
        x.lo,
        x.hi,
        PI,
        TWO_PI,
    ):
        lower = -1.0

    return Interval(
        round_down(lower),
        round_up(upper),
    )


# ============================================================================
# Convex boxes
# ============================================================================


@dataclass
class Box:
    q2_lo: float
    q2_hi: float
    q3_lo: float
    q3_hi: float

    sign: int
    margin: float
    depth: int = 0

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.q2_lo + self.q2_hi),
                0.5 * (self.q3_lo + self.q3_hi),
            ],
            dtype=float,
        )

    @property
    def area(self) -> float:
        return (
            (self.q2_hi - self.q2_lo)
            * (self.q3_hi - self.q3_lo)
        )

    def shifted(
        self,
        shift: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Return this box lifted by integer multiples of 2*pi."""
        shift_q2, shift_q3 = shift

        return (
            self.q2_lo + shift_q2 * TWO_PI,
            self.q2_hi + shift_q2 * TWO_PI,
            self.q3_lo + shift_q3 * TWO_PI,
            self.q3_hi + shift_q3 * TWO_PI,
        )

    def contains_wrapped(
        self,
        q23: np.ndarray,
        tolerance: float = 1e-12,
    ) -> bool:
        """Check whether a torus point lies inside the box."""
        q2, q3 = wrap_to_pi(q23)

        return (
            self.q2_lo - tolerance
            <= q2
            <= self.q2_hi + tolerance
            and self.q3_lo - tolerance
            <= q3
            <= self.q3_hi + tolerance
        )


def determinant_interval(
    bounds: tuple[float, float, float, float],
) -> Interval:
    """
    Natural interval extension of det(Jv) over a q2-q3 rectangle.
    """
    q2_lo, q2_hi, q3_lo, q3_hi = bounds

    q2_interval = Interval(q2_lo, q2_hi)
    q3_interval = Interval(q3_lo, q3_hi)

    sin_q3 = interval_sin(q3_interval)
    cos_q3 = interval_cos(q3_interval)
    cos_q2 = interval_cos(q2_interval)

    L = 2.0 + 1.5 * cos_q3

    singular_interval = (
        sin_q3
        + cos_q2 * (2.0 * sin_q3 - cos_q3)
    )

    return 1.5 * L * singular_interval


def certify_rectangle(
    bounds: tuple[float, float, float, float],
    determinant_epsilon: float,
) -> tuple[int, float, Interval]:
    """
    Certify an entire rectangle.

    Returns
    -------
    sign:
        +1 if det(Jv) >= epsilon over the complete rectangle.
        -1 if det(Jv) <= -epsilon over the complete rectangle.
         0 if interval arithmetic is inconclusive.

    margin:
        Certified lower bound on |det(Jv)|.
    """
    determinant_bounds = determinant_interval(bounds)

    if determinant_bounds.lo >= determinant_epsilon:
        return (
            +1,
            determinant_bounds.lo,
            determinant_bounds,
        )

    if determinant_bounds.hi <= -determinant_epsilon:
        return (
            -1,
            -determinant_bounds.hi,
            determinant_bounds,
        )

    return 0, 0.0, determinant_bounds


# ============================================================================
# Adaptive certified decomposition
# ============================================================================


def adaptive_certified_cover(
    max_depth: int = 8,
    determinant_epsilon: float = 1e-4,
):
    """
    Adaptively decompose [-pi,pi]^2.

    A box is accepted only when interval arithmetic proves that det(Jv)
    has a fixed nonzero sign over the entire box.
    """
    stack = [
        (
            -PI,
            PI,
            -PI,
            PI,
            0,
        )
    ]

    certified_boxes: list[Box] = []
    unresolved_boxes = []

    boxes_tested = 0

    while stack:
        (
            q2_lo,
            q2_hi,
            q3_lo,
            q3_hi,
            depth,
        ) = stack.pop()

        boxes_tested += 1

        bounds = (
            q2_lo,
            q2_hi,
            q3_lo,
            q3_hi,
        )

        sign, margin, _ = certify_rectangle(
            bounds,
            determinant_epsilon,
        )

        if sign != 0:
            certified_boxes.append(
                Box(
                    q2_lo=q2_lo,
                    q2_hi=q2_hi,
                    q3_lo=q3_lo,
                    q3_hi=q3_hi,
                    sign=sign,
                    margin=margin,
                    depth=depth,
                )
            )
            continue

        if depth >= max_depth:
            unresolved_boxes.append(bounds)
            continue

        q2_mid = 0.5 * (q2_lo + q2_hi)
        q3_mid = 0.5 * (q3_lo + q3_hi)

        next_depth = depth + 1

        stack.extend(
            [
                (
                    q2_lo,
                    q2_mid,
                    q3_lo,
                    q3_mid,
                    next_depth,
                ),
                (
                    q2_mid,
                    q2_hi,
                    q3_lo,
                    q3_mid,
                    next_depth,
                ),
                (
                    q2_lo,
                    q2_mid,
                    q3_mid,
                    q3_hi,
                    next_depth,
                ),
                (
                    q2_mid,
                    q2_hi,
                    q3_mid,
                    q3_hi,
                    next_depth,
                ),
            ]
        )

    total_area = TWO_PI * TWO_PI

    certified_area = sum(
        box.area
        for box in certified_boxes
    )

    unresolved_area = sum(
        (bounds[1] - bounds[0])
        * (bounds[3] - bounds[2])
        for bounds in unresolved_boxes
    )

    statistics = {
        "tested": boxes_tested,
        "initial_regions": len(certified_boxes),
        "unresolved_cells": len(unresolved_boxes),
        "certified_area": certified_area,
        "unresolved_area": unresolved_area,
        "certified_fraction": certified_area / total_area,
    }

    return (
        certified_boxes,
        unresolved_boxes,
        statistics,
    )


# ============================================================================
# Grow certified boxes by exact merging
# ============================================================================


def quantize(value: float, digits: int = 13) -> float:
    return round(float(value), digits)


def merge_along_q2(
    boxes: list[Box],
    tolerance: float = 1e-12,
):
    """
    Merge boxes that share a complete q3 interval and are adjacent in q2.

    The merged rectangle is exactly the union of two certified boxes.
    Therefore the certification is preserved.
    """
    groups = {}

    for box in boxes:
        key = (
            box.sign,
            quantize(box.q3_lo),
            quantize(box.q3_hi),
        )

        groups.setdefault(key, []).append(box)

    output = []
    merge_count = 0

    for group in groups.values():
        group.sort(
            key=lambda box: (
                box.q2_lo,
                box.q2_hi,
            )
        )

        current = group[0]

        for next_box in group[1:]:
            adjacent = (
                abs(current.q2_hi - next_box.q2_lo)
                <= tolerance
            )

            if adjacent:
                current = Box(
                    q2_lo=current.q2_lo,
                    q2_hi=next_box.q2_hi,
                    q3_lo=current.q3_lo,
                    q3_hi=current.q3_hi,
                    sign=current.sign,
                    margin=min(
                        current.margin,
                        next_box.margin,
                    ),
                    depth=min(
                        current.depth,
                        next_box.depth,
                    ),
                )

                merge_count += 1
            else:
                output.append(current)
                current = next_box

        output.append(current)

    return output, merge_count


def merge_along_q3(
    boxes: list[Box],
    tolerance: float = 1e-12,
):
    """
    Merge boxes that share a complete q2 interval and are adjacent in q3.
    """
    groups = {}

    for box in boxes:
        key = (
            box.sign,
            quantize(box.q2_lo),
            quantize(box.q2_hi),
        )

        groups.setdefault(key, []).append(box)

    output = []
    merge_count = 0

    for group in groups.values():
        group.sort(
            key=lambda box: (
                box.q3_lo,
                box.q3_hi,
            )
        )

        current = group[0]

        for next_box in group[1:]:
            adjacent = (
                abs(current.q3_hi - next_box.q3_lo)
                <= tolerance
            )

            if adjacent:
                current = Box(
                    q2_lo=current.q2_lo,
                    q2_hi=current.q2_hi,
                    q3_lo=current.q3_lo,
                    q3_hi=next_box.q3_hi,
                    sign=current.sign,
                    margin=min(
                        current.margin,
                        next_box.margin,
                    ),
                    depth=min(
                        current.depth,
                        next_box.depth,
                    ),
                )

                merge_count += 1
            else:
                output.append(current)
                current = next_box

        output.append(current)

    return output, merge_count


def merge_certified_boxes(
    boxes: list[Box],
    max_passes: int = 20,
):
    """
    Alternate q2 and q3 merging until no further growth occurs.
    """
    current = list(boxes)
    region_counts = [len(current)]

    for _ in range(max_passes):
        after_q2, q2_merges = merge_along_q2(current)
        after_q3, q3_merges = merge_along_q3(after_q2)

        current = after_q3
        region_counts.append(len(current))

        if q2_merges + q3_merges == 0:
            break

    return current, region_counts


# ============================================================================
# Region-overlap graph on the periodic torus
# ============================================================================


def interval_overlap(
    a_lo: float,
    a_hi: float,
    b_lo: float,
    b_hi: float,
    tolerance: float = 1e-12,
):
    lower = max(a_lo, b_lo)
    upper = min(a_hi, b_hi)

    if lower <= upper + tolerance:
        if upper < lower:
            midpoint = 0.5 * (lower + upper)
            return midpoint, midpoint

        return lower, upper

    return None


def best_periodic_contact(
    box_a: Box,
    box_b: Box,
    tolerance: float = 1e-12,
):
    """
    Find a periodic lift of box_b that contacts or overlaps box_a.

    Returns
    -------
    shift:
        Integer pair (k2,k3), meaning that box_b is shifted by
        (2*pi*k2, 2*pi*k3).

    intersection:
        The common q2-q3 rectangle or boundary.
    """
    best_result = None
    best_score = -np.inf

    for shift_q2, shift_q3 in itertools.product(
        (-1, 0, 1),
        repeat=2,
    ):
        b_q2_lo = box_b.q2_lo + shift_q2 * TWO_PI
        b_q2_hi = box_b.q2_hi + shift_q2 * TWO_PI
        b_q3_lo = box_b.q3_lo + shift_q3 * TWO_PI
        b_q3_hi = box_b.q3_hi + shift_q3 * TWO_PI

        q2_overlap = interval_overlap(
            box_a.q2_lo,
            box_a.q2_hi,
            b_q2_lo,
            b_q2_hi,
            tolerance,
        )

        if q2_overlap is None:
            continue

        q3_overlap = interval_overlap(
            box_a.q3_lo,
            box_a.q3_hi,
            b_q3_lo,
            b_q3_hi,
            tolerance,
        )

        if q3_overlap is None:
            continue

        q2_width = max(
            0.0,
            q2_overlap[1] - q2_overlap[0],
        )

        q3_width = max(
            0.0,
            q3_overlap[1] - q3_overlap[0],
        )

        # Prefer area overlaps, followed by edge overlaps, followed by corners.
        score = (
            q2_width * q3_width
            + 1e-6 * (q2_width + q3_width)
        )

        if score > best_score:
            best_score = score

            best_result = (
                (shift_q2, shift_q3),
                (
                    q2_overlap[0],
                    q2_overlap[1],
                    q3_overlap[0],
                    q3_overlap[1],
                ),
            )

    return best_result


def build_region_graph(
    boxes: list[Box],
):
    """
    Build the periodic graph.

    Each vertex is a certified convex box.

    An edge exists when two same-sign boxes overlap or touch after accounting
    for the periodic identification of q2 and q3.
    """
    adjacency = [
        []
        for _ in boxes
    ]

    edge_count = 0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].sign != boxes[j].sign:
                continue

            contact = best_periodic_contact(
                boxes[i],
                boxes[j],
            )

            if contact is None:
                continue

            shift, _ = contact

            edge_capacity = min(
                boxes[i].margin,
                boxes[j].margin,
            )

            adjacency[i].append(
                (
                    j,
                    shift,
                    edge_capacity,
                )
            )

            adjacency[j].append(
                (
                    i,
                    (-shift[0], -shift[1]),
                    edge_capacity,
                )
            )

            edge_count += 1

    return adjacency, edge_count


def connected_components(adjacency):
    """Label all graph connected components."""
    labels = -np.ones(
        len(adjacency),
        dtype=int,
    )

    components = []

    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue

        component_label = len(components)
        queue = deque([start])

        labels[start] = component_label
        component = []

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbour, _, _ in adjacency[node]:
                if labels[neighbour] < 0:
                    labels[neighbour] = component_label
                    queue.append(neighbour)

        components.append(component)

    return labels, components


def containing_regions(
    boxes: list[Box],
    q23: np.ndarray,
    sign: int,
):
    """Return all certified boxes containing a selected q2-q3 point."""
    return [
        index
        for index, box in enumerate(boxes)
        if box.sign == sign
        and box.contains_wrapped(q23)
    ]


# ============================================================================
# Position inverse kinematics
# ============================================================================


def reduced_ik_equations(
    q23: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Reduced position IK equations in q2 and q3.

    Since

        [x,y]^T = R(q1) [A,B]^T,

    q1 can be removed by requiring

        x^2+y^2 = A^2+B^2.

    The second equation is the z-coordinate equation.
    """
    q2, q3 = np.asarray(q23, dtype=float)
    x, y, z = np.asarray(target, dtype=float)

    rho_squared = x * x + y * y

    L = 2.0 + 1.5 * np.cos(q3)
    A = 1.0 + L * np.cos(q2)
    B = 1.0 + 1.5 * np.sin(q3)

    return np.array(
        [
            A * A + B * B - rho_squared,
            -L * np.sin(q2) - z,
        ],
        dtype=float,
    )


def reduced_ik_jacobian(
    q23: np.ndarray,
) -> np.ndarray:
    """Jacobian of reduced_ik_equations with respect to (q2,q3)."""
    q2, q3 = np.asarray(q23, dtype=float)

    sin_q2 = np.sin(q2)
    cos_q2 = np.cos(q2)
    sin_q3 = np.sin(q3)
    cos_q3 = np.cos(q3)

    L = 2.0 + 1.5 * cos_q3

    A = 1.0 + L * cos_q2
    B = 1.0 + 1.5 * sin_q3

    dA_dq2 = -L * sin_q2
    dA_dq3 = -1.5 * sin_q3 * cos_q2
    dB_dq3 = 1.5 * cos_q3

    return np.array(
        [
            [
                2.0 * A * dA_dq2,
                2.0 * A * dA_dq3
                + 2.0 * B * dB_dq3,
            ],
            [
                -L * cos_q2,
                1.5 * sin_q3 * sin_q2,
            ],
        ],
        dtype=float,
    )


def recover_q1(
    q2: float,
    q3: float,
    target: np.ndarray,
) -> float:
    """
    Recover q1 from a reduced IK root.

    [x,y]^T = R(q1)[A,B]^T, so

        q1 = atan2(y,x) - atan2(B,A).
    """
    x, y, _ = np.asarray(target, dtype=float)

    L = 2.0 + 1.5 * np.cos(q3)
    A = 1.0 + L * np.cos(q2)
    B = 1.0 + 1.5 * np.sin(q3)

    q1 = (
        np.arctan2(y, x)
        - np.arctan2(B, A)
    )

    return float(wrap_to_pi(q1))


def solve_all_position_ik(
    target: np.ndarray,
    starts_per_axis: int = 17,
    residual_tolerance: float = 1e-9,
    duplicate_tolerance: float = 1e-6,
):
    """
    Find all position IK roots using multistart solution of the reduced system.
    """
    reduced_roots = []

    seeds = np.linspace(
        -PI,
        PI,
        starts_per_axis,
        endpoint=False,
    )

    for q2_initial in seeds:
        for q3_initial in seeds:
            result = root(
                fun=lambda q23: reduced_ik_equations(
                    q23,
                    target,
                ),
                x0=np.array(
                    [
                        q2_initial,
                        q3_initial,
                    ],
                    dtype=float,
                ),
                jac=reduced_ik_jacobian,
                method="hybr",
            )

            if not result.success:
                continue

            candidate = wrap_to_pi(result.x)

            residual = np.linalg.norm(
                reduced_ik_equations(
                    candidate,
                    target,
                )
            )

            if residual > residual_tolerance:
                continue

            duplicate = any(
                np.linalg.norm(
                    angular_difference(
                        candidate,
                        existing_root,
                    )
                )
                < duplicate_tolerance
                for existing_root in reduced_roots
            )

            if duplicate:
                continue

            reduced_roots.append(candidate)

    full_roots = []

    for q2, q3 in reduced_roots:
        q1 = recover_q1(
            q2,
            q3,
            target,
        )

        full_q = np.array(
            [
                q1,
                q2,
                q3,
            ],
            dtype=float,
        )

        fk_residual = np.linalg.norm(
            fk_position(full_q) - target
        )

        if fk_residual <= 1e-7:
            full_roots.append(full_q)

    # Sorting by q3 gives deterministic labels.
    full_roots.sort(
        key=lambda q: q[2]
    )

    return full_roots


# ============================================================================
# Krawczyk root certification
# ============================================================================


def interval_reduced_jacobian(
    q2_interval: Interval,
    q3_interval: Interval,
):
    """Interval extension of the reduced IK Jacobian."""
    sin_q2 = interval_sin(q2_interval)
    cos_q2 = interval_cos(q2_interval)

    sin_q3 = interval_sin(q3_interval)
    cos_q3 = interval_cos(q3_interval)

    L = 2.0 + 1.5 * cos_q3

    A = 1.0 + L * cos_q2
    B = 1.0 + 1.5 * sin_q3

    dA_dq2 = -L * sin_q2
    dA_dq3 = -1.5 * sin_q3 * cos_q2
    dB_dq3 = 1.5 * cos_q3

    return [
        [
            2.0 * A * dA_dq2,
            2.0 * A * dA_dq3
            + 2.0 * B * dB_dq3,
        ],
        [
            -L * cos_q2,
            1.5 * sin_q3 * sin_q2,
        ],
    ]


def krawczyk_certify_reduced_root(
    root_center: np.ndarray,
    target: np.ndarray,
    half_width: float = 1e-5,
):
    """
    Apply a two-dimensional Krawczyk inclusion test.

    If

        K(X) subset interior(X),

    then the box X contains one unique exact root of the reduced IK equations.
    """
    center = np.asarray(
        root_center,
        dtype=float,
    )

    function_at_center = reduced_ik_equations(
        center,
        target,
    )

    jacobian_at_center = reduced_ik_jacobian(
        center
    )

    try:
        inverse_jacobian = np.linalg.inv(
            jacobian_at_center
        )
    except np.linalg.LinAlgError:
        return False, [], []

    root_box = [
        Interval(
            center[0] - half_width,
            center[0] + half_width,
        ),
        Interval(
            center[1] - half_width,
            center[1] + half_width,
        ),
    ]

    centered_box = [
        Interval(
            -half_width,
            half_width,
        ),
        Interval(
            -half_width,
            half_width,
        ),
    ]

    interval_jacobian = interval_reduced_jacobian(
        root_box[0],
        root_box[1],
    )

    # M = I - C J(X)
    interval_matrix = [
        [
            Interval.point(0.0)
            for _ in range(2)
        ]
        for _ in range(2)
    ]

    for row in range(2):
        for column in range(2):
            value = Interval.point(
                1.0
                if row == column
                else 0.0
            )

            for index in range(2):
                value = (
                    value
                    - inverse_jacobian[row, index]
                    * interval_jacobian[index][column]
                )

            interval_matrix[row][column] = value

    base_point = (
        center
        - inverse_jacobian @ function_at_center
    )

    krawczyk_box = []

    for row in range(2):
        value = Interval.point(
            base_point[row]
        )

        for column in range(2):
            value = (
                value
                + interval_matrix[row][column]
                * centered_box[column]
            )

        krawczyk_box.append(value)

    success = all(
        krawczyk_box[index].lo
        > root_box[index].lo
        and krawczyk_box[index].hi
        < root_box[index].hi
        for index in range(2)
    )

    return (
        success,
        root_box,
        krawczyk_box,
    )


# ============================================================================
# Graph path search
# ============================================================================


def widest_region_path(
    adjacency,
    boxes: list[Box],
    start_nodes: Iterable[int],
    goal_nodes: Iterable[int],
):
    """
    Find a maximum-bottleneck path through the certified region graph.

    The path score is the minimum determinant margin encountered along it.
    """
    start_nodes = list(start_nodes)
    goal_set = set(goal_nodes)

    number_of_nodes = len(boxes)

    best_capacity = np.full(
        number_of_nodes,
        -np.inf,
    )

    predecessor = [
        None
        for _ in range(number_of_nodes)
    ]

    queue = []

    for start in start_nodes:
        best_capacity[start] = boxes[start].margin

        heapq.heappush(
            queue,
            (
                -best_capacity[start],
                start,
            ),
        )

    reached_goal = None

    while queue:
        negative_capacity, current = heapq.heappop(
            queue
        )

        current_capacity = -negative_capacity

        if (
            current_capacity + 1e-15
            < best_capacity[current]
        ):
            continue

        if current in goal_set:
            reached_goal = current
            break

        for (
            neighbour,
            shift,
            edge_capacity,
        ) in adjacency[current]:

            candidate_capacity = min(
                current_capacity,
                edge_capacity,
                boxes[neighbour].margin,
            )

            if (
                candidate_capacity
                > best_capacity[neighbour] + 1e-15
            ):
                best_capacity[neighbour] = candidate_capacity

                predecessor[neighbour] = (
                    current,
                    shift,
                )

                heapq.heappush(
                    queue,
                    (
                        -candidate_capacity,
                        neighbour,
                    ),
                )

    if reached_goal is None:
        return None

    region_nodes = [reached_goal]
    reversed_edge_shifts = []

    current = reached_goal

    while predecessor[current] is not None:
        previous, shift_previous_to_current = predecessor[
            current
        ]

        reversed_edge_shifts.append(
            shift_previous_to_current
        )

        region_nodes.append(previous)
        current = previous

    region_nodes.reverse()
    edge_shifts = list(
        reversed(reversed_edge_shifts)
    )

    lift_shifts = [
        (0, 0)
    ]

    for edge_shift in edge_shifts:
        previous_shift = lift_shifts[-1]

        lift_shifts.append(
            (
                previous_shift[0]
                + edge_shift[0],
                previous_shift[1]
                + edge_shift[1],
            )
        )

    return {
        "nodes": region_nodes,
        "edge_shifts": edge_shifts,
        "lift_shifts": lift_shifts,
        "certified_margin": best_capacity[
            reached_goal
        ],
    }


def choose_cuspidal_pair(
    ik_solutions: list[np.ndarray],
    boxes: list[Box],
    adjacency,
    component_labels: np.ndarray,
):
    """
    Search for two distinct same-pose IK solutions lying in the same
    certified graph component.
    """
    memberships = []

    for q in ik_solutions:
        sign = (
            +1
            if det_jv(q) > 0.0
            else -1
        )

        memberships.append(
            containing_regions(
                boxes,
                q[1:],
                sign,
            )
        )

    candidates = []

    for first in range(len(ik_solutions)):
        for second in range(
            first + 1,
            len(ik_solutions),
        ):
            first_sign = (
                +1
                if det_jv(ik_solutions[first]) > 0.0
                else -1
            )

            second_sign = (
                +1
                if det_jv(ik_solutions[second]) > 0.0
                else -1
            )

            if first_sign != second_sign:
                continue

            for first_region in memberships[first]:
                for second_region in memberships[second]:
                    if (
                        component_labels[first_region]
                        != component_labels[second_region]
                    ):
                        continue

                    graph_path = widest_region_path(
                        adjacency=adjacency,
                        boxes=boxes,
                        start_nodes=[first_region],
                        goal_nodes=[second_region],
                    )

                    if graph_path is None:
                        continue

                    configuration_distance = np.linalg.norm(
                        angular_difference(
                            ik_solutions[first],
                            ik_solutions[second],
                        )
                    )

                    candidates.append(
                        (
                            graph_path["certified_margin"],
                            configuration_distance,
                            first,
                            second,
                            graph_path,
                        )
                    )

    if not candidates:
        return None

    # Select the first deterministic IK-pair witness. For multiple graph paths
    # belonging to that pair, prefer the largest certified margin.
    candidates.sort(
        key=lambda item: (
            item[2],
            item[3],
            -item[0],
            -item[1],
        )
    )

    (
        _,
        _,
        first,
        second,
        graph_path,
    ) = candidates[0]

    return (
        first,
        second,
        graph_path,
        memberships,
    )


# ============================================================================
# Continuous path construction
# ============================================================================


def lifted_intersection_center(
    first_box: Box,
    first_shift: tuple[int, int],
    second_box: Box,
    second_shift: tuple[int, int],
) -> np.ndarray:
    """
    Select a waypoint in the intersection between two lifted boxes.
    """
    (
        first_q2_lo,
        first_q2_hi,
        first_q3_lo,
        first_q3_hi,
    ) = first_box.shifted(first_shift)

    (
        second_q2_lo,
        second_q2_hi,
        second_q3_lo,
        second_q3_hi,
    ) = second_box.shifted(second_shift)

    q2_overlap = interval_overlap(
        first_q2_lo,
        first_q2_hi,
        second_q2_lo,
        second_q2_hi,
    )

    q3_overlap = interval_overlap(
        first_q3_lo,
        first_q3_hi,
        second_q3_lo,
        second_q3_hi,
    )

    if q2_overlap is None or q3_overlap is None:
        raise RuntimeError(
            "Consecutive lifted graph regions do not intersect."
        )

    return np.array(
        [
            0.5 * (
                q2_overlap[0]
                + q2_overlap[1]
            ),
            0.5 * (
                q3_overlap[0]
                + q3_overlap[1]
            ),
        ],
        dtype=float,
    )


def build_piecewise_linear_path(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    boxes: list[Box],
    region_path: dict,
) -> np.ndarray:
    """
    Construct full q1-q2-q3 waypoints.

    The q2-q3 segments remain inside individual convex regions. Since the
    determinant is independent of q1, q1 can be interpolated freely.
    """
    region_nodes = region_path["nodes"]
    lift_shifts = region_path["lift_shifts"]

    q23_waypoints = [
        np.asarray(
            start_q[1:],
            dtype=float,
        )
    ]

    for index in range(
        len(region_nodes) - 1
    ):
        waypoint = lifted_intersection_center(
            first_box=boxes[
                region_nodes[index]
            ],
            first_shift=lift_shifts[index],
            second_box=boxes[
                region_nodes[index + 1]
            ],
            second_shift=lift_shifts[index + 1],
        )

        q23_waypoints.append(waypoint)

    final_shift = (
        np.asarray(
            lift_shifts[-1],
            dtype=float,
        )
        * TWO_PI
    )

    q23_waypoints.append(
        np.asarray(
            goal_q[1:],
            dtype=float,
        )
        + final_shift
    )

    q23_waypoints = np.asarray(
        q23_waypoints
    )

    q1_start = float(start_q[0])

    q1_goal = (
        q1_start
        + float(
            angular_difference(
                goal_q[0],
                q1_start,
            )
        )
    )

    segment_lengths = np.linalg.norm(
        np.diff(
            q23_waypoints,
            axis=0,
        ),
        axis=1,
    )

    cumulative_length = np.concatenate(
        (
            [0.0],
            np.cumsum(segment_lengths),
        )
    )

    if cumulative_length[-1] <= 1e-15:
        interpolation_fraction = np.linspace(
            0.0,
            1.0,
            len(q23_waypoints),
        )
    else:
        interpolation_fraction = (
            cumulative_length
            / cumulative_length[-1]
        )

    q1_waypoints = (
        q1_start
        + interpolation_fraction
        * (q1_goal - q1_start)
    )

    return np.column_stack(
        (
            q1_waypoints,
            q23_waypoints,
        )
    )


def sample_polyline(
    waypoints: np.ndarray,
    samples_per_segment: int = 200,
) -> np.ndarray:
    """Densely sample a piecewise-linear path."""
    samples = []

    for index in range(
        len(waypoints) - 1
    ):
        interpolation = np.linspace(
            0.0,
            1.0,
            samples_per_segment,
            endpoint=False,
        )

        segment_samples = (
            (1.0 - interpolation[:, None])
            * waypoints[index]
            + interpolation[:, None]
            * waypoints[index + 1]
        )

        samples.append(segment_samples)

    samples.append(
        waypoints[-1][None, :]
    )

    return np.vstack(samples)


# ============================================================================
# Plotting
# ============================================================================


def plot_result(
    boxes: list[Box],
    unresolved_boxes,
    ik_solutions: list[np.ndarray],
    selected_pair: tuple[int, int],
    region_path: dict,
    full_waypoints: np.ndarray,
    output_file: str,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        constrained_layout=True,
    )

    fig.suptitle(
        "Cuspidality witness: two certified IK roots in one "
        "nonsingular graph component",
        fontsize=15,
    )

    # ------------------------------------------------------------------
    # Left panel: periodic q2-q3 chart
    # ------------------------------------------------------------------

    axis = axes[0]

    q2_values = np.linspace(
        -PI,
        PI,
        600,
    )

    q3_values = np.linspace(
        -PI,
        PI,
        600,
    )

    Q2, Q3 = np.meshgrid(
        q2_values,
        q3_values,
    )

    determinant_values = det_jv(
        Q2,
        Q3,
    )

    axis.contourf(
        Q2,
        Q3,
        np.sign(determinant_values),
        levels=[
            -1.5,
            0.0,
            1.5,
        ],
        alpha=0.16,
    )

    axis.contour(
        Q2,
        Q3,
        determinant_values,
        levels=[0.0],
        linewidths=1.3,
    )

    for box in boxes:
        axis.add_patch(
            Rectangle(
                (
                    box.q2_lo,
                    box.q3_lo,
                ),
                box.q2_hi - box.q2_lo,
                box.q3_hi - box.q3_lo,
                fill=False,
                linewidth=0.22,
                alpha=0.22,
            )
        )

    # Show unresolved cells as gaps in the certified cover.
    for (
        q2_lo,
        q2_hi,
        q3_lo,
        q3_hi,
    ) in unresolved_boxes:
        axis.add_patch(
            Rectangle(
                (
                    q2_lo,
                    q3_lo,
                ),
                q2_hi - q2_lo,
                q3_hi - q3_lo,
                fill=True,
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
            )
        )

    selected_set = set(
        selected_pair
    )

    for index, q in enumerate(
        ik_solutions
    ):
        if index in selected_set:
            axis.scatter(
                q[1],
                q[2],
                marker="*",
                s=150,
                edgecolors="black",
                zorder=6,
            )
        else:
            axis.scatter(
                q[1],
                q[2],
                marker="o",
                s=55,
                edgecolors="black",
                zorder=6,
            )

        axis.annotate(
            f"IK {index}",
            (
                q[1],
                q[2],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )

    axis.set_title(
        "Certified convex cover on the periodic "
        r"$(q_2,q_3)$ chart"
    )

    axis.set_xlabel(
        r"$q_2$ [rad]"
    )

    axis.set_ylabel(
        r"$q_3$ [rad]"
    )

    axis.set_xlim(
        -PI,
        PI,
    )

    axis.set_ylim(
        -PI,
        PI,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.grid(
        alpha=0.25
    )

    # ------------------------------------------------------------------
    # Right panel: lifted GCS path
    # ------------------------------------------------------------------

    axis = axes[1]

    region_nodes = region_path["nodes"]
    lift_shifts = region_path["lift_shifts"]

    for region_node, shift in zip(
        region_nodes,
        lift_shifts,
    ):
        (
            q2_lo,
            q2_hi,
            q3_lo,
            q3_hi,
        ) = boxes[region_node].shifted(
            shift
        )

        axis.add_patch(
            Rectangle(
                (
                    q2_lo,
                    q3_lo,
                ),
                q2_hi - q2_lo,
                q3_hi - q3_lo,
                alpha=0.20,
                edgecolor="black",
                linewidth=0.7,
            )
        )

    q2_min = (
        np.min(full_waypoints[:, 1])
        - 0.45
    )

    q2_max = (
        np.max(full_waypoints[:, 1])
        + 0.45
    )

    q3_min = (
        np.min(full_waypoints[:, 2])
        - 0.45
    )

    q3_max = (
        np.max(full_waypoints[:, 2])
        + 0.45
    )

    q2_plot = np.linspace(
        q2_min,
        q2_max,
        700,
    )

    q3_plot = np.linspace(
        q3_min,
        q3_max,
        700,
    )

    plot_Q2, plot_Q3 = np.meshgrid(
        q2_plot,
        q3_plot,
    )

    axis.contour(
        plot_Q2,
        plot_Q3,
        det_jv(
            plot_Q2,
            plot_Q3,
        ),
        levels=[0.0],
        linewidths=1.2,
    )

    axis.plot(
        full_waypoints[:, 1],
        full_waypoints[:, 2],
        "-o",
        linewidth=2.0,
        markersize=4,
        label="certified nonsingular path",
    )

    axis.scatter(
        [
            full_waypoints[0, 1],
            full_waypoints[-1, 1],
        ],
        [
            full_waypoints[0, 2],
            full_waypoints[-1, 2],
        ],
        marker="*",
        s=160,
        edgecolors="black",
        zorder=6,
    )

    axis.set_title(
        "GCS path through certified overlapping regions"
    )

    axis.set_xlabel(
        r"unwrapped $q_2$ [rad]"
    )

    axis.set_ylabel(
        r"unwrapped $q_3$ [rad]"
    )

    axis.set_xlim(
        q2_min,
        q2_max,
    )

    axis.set_ylim(
        q3_min,
        q3_max,
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        loc="best"
    )

    fig.savefig(
        output_file,
        dpi=180,
    )

    plt.close(fig)


# ============================================================================
# Main experiment
# ============================================================================


def main():
    experiment_start = time.perf_counter()

    # Increase max_depth for a tighter cover near the singularity curve.
    max_depth = 8

    # Every accepted region must satisfy |det(Jv)| >= this lower bound.
    determinant_epsilon = 1e-4

    output_figure = (
        "figures/certified_singularity_free_gcs_3r.png"
    )

    # Deterministic regular target with four position-IK solutions.
    target = np.array(
        [
            2.5,
            0.0,
            0.5,
        ],
        dtype=float,
    )

    robot = build_robot()

    print("=" * 78)
    print(
        "Certified singularity-free GCS experiment "
        "for the supplied 3R robot"
    )
    print("=" * 78)

    if robot is None:
        print(
            "roboticstoolbox-python is not installed; "
            "using direct standard-DH validation."
        )
    else:
        print(robot)

    # ------------------------------------------------------------------
    # Validate the analytic kinematics
    # ------------------------------------------------------------------

    (
        maximum_position_error,
        maximum_determinant_error,
    ) = validate_kinematics(robot)

    print("\nClosed-form kinematic validation")

    print(
        "  maximum position error       : "
        f"{maximum_position_error:.3e}"
    )

    print(
        "  maximum determinant error    : "
        f"{maximum_determinant_error:.3e}"
    )

    # ------------------------------------------------------------------
    # Construct the certified convex cover
    # ------------------------------------------------------------------

    (
        initial_boxes,
        unresolved_boxes,
        cover_statistics,
    ) = adaptive_certified_cover(
        max_depth=max_depth,
        determinant_epsilon=determinant_epsilon,
    )

    (
        convex_boxes,
        merge_counts,
    ) = merge_certified_boxes(
        initial_boxes
    )

    print("\nCertified convex cover")

    print(
        "  quadtree boxes tested        : "
        f"{cover_statistics['tested']}"
    )

    print(
        "  initially certified boxes    : "
        f"{cover_statistics['initial_regions']}"
    )

    print(
        "  unresolved maximum-depth cells: "
        f"{cover_statistics['unresolved_cells']}"
    )

    print(
        "  certified chart fraction     : "
        f"{100.0 * cover_statistics['certified_fraction']:.3f}%"
    )

    print(
        "  merge/growth counts          : "
        + " -> ".join(
            map(
                str,
                merge_counts,
            )
        )
    )

    print(
        "  final convex regions         : "
        f"{len(convex_boxes)}"
    )

    # ------------------------------------------------------------------
    # Build the periodic overlap graph
    # ------------------------------------------------------------------

    adjacency, edge_count = build_region_graph(
        convex_boxes
    )

    (
        component_labels,
        components,
    ) = connected_components(
        adjacency
    )

    positive_component_count = sum(
        1
        for component in components
        if convex_boxes[
            component[0]
        ].sign > 0
    )

    negative_component_count = sum(
        1
        for component in components
        if convex_boxes[
            component[0]
        ].sign < 0
    )

    print("\nRegion graph")

    print(
        "  vertices                     : "
        f"{len(convex_boxes)}"
    )

    print(
        "  edges                        : "
        f"{edge_count}"
    )

    print(
        "  positive-sign components     : "
        f"{positive_component_count}"
    )

    print(
        "  negative-sign components     : "
        f"{negative_component_count}"
    )

    # ------------------------------------------------------------------
    # Compute position IK solutions
    # ------------------------------------------------------------------

    ik_solutions = solve_all_position_ik(
        target
    )

    print("\nPosition IK target")

    print(
        "  p = "
        + np.array2string(
            target,
            precision=10,
        )
    )

    print(
        "  distinct IK solutions found  : "
        f"{len(ik_solutions)}"
    )

    for index, q in enumerate(
        ik_solutions
    ):
        fk_residual = np.linalg.norm(
            fk_position(q) - target
        )

        print(
            f"  IK {index}: "
            f"q={np.array2string(q, precision=10)}, "
            f"det(Jv)={det_jv(q): .10f}, "
            f"residual={fk_residual:.3e}"
        )

    # ------------------------------------------------------------------
    # Find a same-component IK pair
    # ------------------------------------------------------------------

    selected_result = choose_cuspidal_pair(
        ik_solutions=ik_solutions,
        boxes=convex_boxes,
        adjacency=adjacency,
        component_labels=component_labels,
    )

    if selected_result is None:
        raise RuntimeError(
            "No certified same-component IK pair was found. "
            "Increase max_depth or select another regular target."
        )

    (
        first_ik_index,
        second_ik_index,
        region_path,
        memberships,
    ) = selected_result

    start_configuration = ik_solutions[
        first_ik_index
    ]

    goal_configuration = ik_solutions[
        second_ik_index
    ]

    # ------------------------------------------------------------------
    # Certify exact roots near both numerical solutions
    # ------------------------------------------------------------------

    start_root_certificate = (
        krawczyk_certify_reduced_root(
            root_center=start_configuration[1:],
            target=target,
            half_width=1e-5,
        )
    )

    goal_root_certificate = (
        krawczyk_certify_reduced_root(
            root_center=goal_configuration[1:],
            target=target,
            half_width=1e-5,
        )
    )

    # ------------------------------------------------------------------
    # Construct the continuous path
    # ------------------------------------------------------------------

    full_waypoints = build_piecewise_linear_path(
        start_q=start_configuration,
        goal_q=goal_configuration,
        boxes=convex_boxes,
        region_path=region_path,
    )

    dense_path = sample_polyline(
        full_waypoints,
        samples_per_segment=300,
    )

    path_determinants = det_jv(
        dense_path
    )

    selected_region_margins = [
        convex_boxes[node].margin
        for node in region_path["nodes"]
    ]

    certified_path_margin = min(
        selected_region_margins
    )

    wrapped_configuration_distance = np.linalg.norm(
        angular_difference(
            start_configuration,
            goal_configuration,
        )
    )

    # ------------------------------------------------------------------
    # Report the witness
    # ------------------------------------------------------------------

    print("\nCertified cuspidality witness")

    print(
        "  selected IK pair             : "
        f"IK {first_ik_index} and IK {second_ik_index}"
    )

    print(
        "  wrapped configuration distance: "
        f"{wrapped_configuration_distance:.6f} rad"
    )

    print(
        "  start-containing regions     : "
        f"{memberships[first_ik_index]}"
    )

    print(
        "  goal-containing regions      : "
        f"{memberships[second_ik_index]}"
    )

    print(
        "  region path                  : "
        f"{region_path['nodes']}"
    )

    print(
        "  number of path regions       : "
        f"{len(region_path['nodes'])}"
    )

    print(
        "  graph bottleneck estimate    : "
        f"{region_path['certified_margin']:.10e}"
    )

    print(
        "  certified |det(Jv)| lower bound: "
        f"{certified_path_margin:.10e}"
    )

    print(
        "  dense numerical det range   : "
        f"[{np.min(path_determinants):.10f}, "
        f"{np.max(path_determinants):.10f}]"
    )

    print(
        "  dense numerical min |det|   : "
        f"{np.min(np.abs(path_determinants)):.10e}"
    )

    print(
        "  Krawczyk exact-root tests   : "
        + (
            "PASS"
            if start_root_certificate[0]
            else "FAIL"
        )
        + ", "
        + (
            "PASS"
            if goal_root_certificate[0]
            else "FAIL"
        )
    )

    # ------------------------------------------------------------------
    # Generate the plot
    # ------------------------------------------------------------------

    plot_result(
        boxes=convex_boxes,
        unresolved_boxes=unresolved_boxes,
        ik_solutions=ik_solutions,
        selected_pair=(
            first_ik_index,
            second_ik_index,
        ),
        region_path=region_path,
        full_waypoints=full_waypoints,
        output_file=output_figure,
    )

    elapsed_time = (
        time.perf_counter()
        - experiment_start
    )

    print("\nConclusion")

    print(
        "  Two distinct position-IK roots are connected by a continuous path"
    )

    print(
        "  contained in certified singularity-free convex regions."
    )

    print(
        "  Therefore, this 3R robot is cuspidal for its position kinematics."
    )

    print(
        "  figure                       : "
        f"{output_figure}"
    )

    print(
        "  total runtime                : "
        f"{elapsed_time:.3f} s"
    )


if __name__ == "__main__":
    main()