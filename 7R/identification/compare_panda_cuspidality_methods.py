#!/usr/bin/env python3
"""
Compare two positive-witness methods for cuspidality of the Franka Panda.

The implementation follows the reduction in Salunkhe, Gupta and Billard
(2025):

* the Panda is treated as a spherical-shoulder 7R robot;
* joint 5 is the redundant parameter theta_r;
* for each fixed theta_r, the inverted 4R position chain is replaced by the
  standard-DH 3R chain in Table V of the paper.

The two compared methods are:

A. Random-target + linear-path method
   1. Generate a reachable target from a random reduced-3R configuration.
   2. Numerically collect all position-IK roots found by multistart solving.
   3. Select pairs with the same sign of det(J_v).
   4. Test straight joint-space paths, including nearby periodic lifts.

B. Certified-box Graph of Convex Sets (GCS) method
   1. Adaptively cover the (q2,q3) torus with convex boxes whose determinant
      interval excludes zero.
   2. Connect certified boxes that share an edge, including periodic edges.
   3. Precompute connected-component labels.
   4. For the same target sequence and the same IK roots used by method A,
      check whether two distinct roots occur in the same graph component.

The comparison separates GCS offline construction time from online query
cost.  A positive result is a cuspidality witness.  A failure is inconclusive:
multistart IK can miss roots, a straight line can miss a curved connection,
and an incomplete convex cover can miss a graph connection.

Dependencies
------------
Required:
    numpy
    scipy
    matplotlib

For the requested Panda model:
    roboticstoolbox-python

When Robotics Toolbox is unavailable, the script can still run with the
published Panda constants, but it prints a warning.  On the user's machine,
``robot = rtb.models.DH.Panda()`` is used by default.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from itertools import combinations, product
import math
from pathlib import Path
import time
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

try:
    import roboticstoolbox as rtb
except ImportError:  # Allows validation in environments without RTB.
    rtb = None


PI = math.pi
TWO_PI = 2.0 * PI


# =============================================================================
# Panda constants and the paper's parameterised equivalent 3R chain
# =============================================================================


@dataclass(frozen=True)
class PandaPaperConstants:
    """Geometric constants used in Tables IV and V of the paper."""

    d3: float
    a3: float
    d5: float
    a6: float
    d7: float
    source: str


PUBLISHED_CONSTANTS = PandaPaperConstants(
    d3=0.316,
    a3=0.0825,
    d5=0.384,
    a6=0.088,
    d7=0.107,
    source="published Table IV constants",
)


def load_panda_and_constants() -> tuple[object | None, PandaPaperConstants]:
    """
    Instantiate the requested RTB model and extract the constants needed by
    the paper's *original-DH* equivalent chain.

    ``rtb.models.DH.Panda()`` itself uses modified DH.  Consequently the
    original-DH symbols in the paper appear at shifted MDH link indices:

        paper d3 <- robot.links[2].d
        paper a3 <- robot.links[3].a
        paper d5 <- robot.links[4].d
        paper a6 <- robot.links[6].a
        paper d7 <- robot.links[6].d

    The Panda tool transform is intentionally not included.  The paper uses
    the flange offset d7=0.107 m, whereas the RTB model additionally attaches
    a tool transform after the seventh link.
    """
    if rtb is None:
        return None, PUBLISHED_CONSTANTS

    # This is the robot definition requested by the user.
    robot = rtb.models.DH.Panda()

    if robot.n != 7:
        raise ValueError(f"Expected a 7R Panda model, got n={robot.n}.")

    constants = PandaPaperConstants(
        d3=float(robot.links[2].d),
        a3=float(robot.links[3].a),
        d5=float(robot.links[4].d),
        a6=float(robot.links[6].a),
        d7=float(robot.links[6].d),
        source="rtb.models.DH.Panda()",
    )

    expected = np.array(
        [
            PUBLISHED_CONSTANTS.d3,
            PUBLISHED_CONSTANTS.a3,
            PUBLISHED_CONSTANTS.d5,
            PUBLISHED_CONSTANTS.a6,
            PUBLISHED_CONSTANTS.d7,
        ],
        dtype=float,
    )
    extracted = np.array(
        [
            constants.d3,
            constants.a3,
            constants.d5,
            constants.a6,
            constants.d7,
        ],
        dtype=float,
    )

    if not np.allclose(extracted, expected, atol=1e-12, rtol=0.0):
        raise ValueError(
            "The installed DH.Panda model does not contain the geometric "
            "constants expected by the paper.\n"
            f"Extracted: {extracted}\nExpected: {expected}"
        )

    return robot, constants


def wrap_to_pi(angle):
    """Wrap a scalar or array of angles to [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + PI) % TWO_PI - PI


def angular_difference(a, b):
    """Componentwise wrapped difference a-b."""
    return wrap_to_pi(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


@dataclass
class EquivalentPanda3R:
    """Equivalent standard-DH 3R chain for one fixed redundant angle."""

    theta_r: float
    constants: PandaPaperConstants

    def __post_init__(self) -> None:
        self.sin_r = float(np.sin(self.theta_r))
        self.cos_r = float(np.cos(self.theta_r))

        if abs(self.sin_r) < 1e-8:
            raise ValueError(
                "The Table-V mapping is singular at theta_r in {0, +/-pi}. "
                f"Received theta_r={self.theta_r:.12g}."
            )

        self.ell = float(
            np.hypot(self.constants.a3, self.constants.d3)
        )

    def build_rtb_robot(self):
        """Build the Table-V chain explicitly with RTB standard-DH links."""
        if rtb is None:
            return None

        d, a, alpha = self.dh_parameters
        return rtb.DHRobot(
            [
                rtb.RevoluteDH(d=d[index], a=a[index], alpha=alpha[index])
                for index in range(3)
            ],
            name=f"Panda_equivalent_3R_theta_r_{self.theta_r:.6f}",
        )

    @property
    def dh_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (d, a, alpha) from Table V of the paper."""
        c = self.constants
        d = np.array(
            [
                -c.d7,
                c.a3 / self.sin_r,
                c.a3 * self.cos_r / self.sin_r,
            ],
            dtype=float,
        )
        a = np.array(
            [
                -c.a6,
                c.d5,
                -self.ell,
            ],
            dtype=float,
        )
        alpha = np.array(
            [
                -PI / 2.0,
                self.theta_r - PI,
                -PI / 2.0,
            ],
            dtype=float,
        )
        return d, a, alpha

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """Vectorised end-point position of the equivalent 3R chain."""
        q = np.asarray(q, dtype=float)
        q1 = q[..., 0]
        q2 = q[..., 1]
        q3 = q[..., 2]

        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)

        c = self.constants
        h = c.a3 + self.ell * s3
        g = self.ell * c3

        # Position for q1=0.  General q1 rotates [x0,y0] about the z axis.
        x0 = -c.a6 + (c.d5 - g) * c2 - self.cos_r * h * s2
        y0 = self.sin_r * h
        z0 = -c.d7 + (g - c.d5) * s2 - self.cos_r * h * c2

        x = c1 * x0 - s1 * y0
        y = s1 * x0 + c1 * y0

        return np.stack((x, y, z0), axis=-1)

    def det_jv(self, q: np.ndarray) -> np.ndarray:
        """
        Vectorised determinant of the 3x3 translational Jacobian.

        It is independent of q1, as expected for a generic positioning 3R
        chain whose first joint rotates the whole mechanism about the base
        z-axis.
        """
        q = np.asarray(q, dtype=float)
        q2 = q[..., 1]
        q3 = q[..., 2]

        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)

        c = self.constants
        h = c.a3 + self.ell * s3

        u = c.a3 * c3 + c.d5 * s3
        v = (
            self.cos_r * h * s2
            + (self.ell * c3 - c.d5) * c2
        )
        w = c.a6 * (
            c.a3 * c3 * self.cos_r**2
            + c.d5 * s3
            - self.ell * s3 * c3 * self.sin_r**2
        )

        return self.ell * (u * v + w)

    def reduced_equations(self, q23: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Two equations in (q2,q3), after eliminating q1."""
        q2, q3 = np.asarray(q23, dtype=float)
        target = np.asarray(target, dtype=float)

        c = self.constants
        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)

        h = c.a3 + self.ell * s3
        g = self.ell * c3

        x0 = -c.a6 + (c.d5 - g) * c2 - self.cos_r * h * s2
        y0 = self.sin_r * h
        z0 = -c.d7 + (g - c.d5) * s2 - self.cos_r * h * c2

        rho_squared = target[0] ** 2 + target[1] ** 2

        return np.array(
            [
                x0 * x0 + y0 * y0 - rho_squared,
                z0 - target[2],
            ],
            dtype=float,
        )

    def reduced_jacobian(self, q23: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of ``reduced_equations`` with respect to q2,q3."""
        q2, q3 = np.asarray(q23, dtype=float)
        c = self.constants

        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)

        h = c.a3 + self.ell * s3
        g = self.ell * c3

        x0 = -c.a6 + (c.d5 - g) * c2 - self.cos_r * h * s2
        y0 = self.sin_r * h

        dx_dq2 = -(c.d5 - g) * s2 - self.cos_r * h * c2
        dz_dq2 = (g - c.d5) * c2 + self.cos_r * h * s2

        dx_dq3 = self.ell * s3 * c2 - self.cos_r * g * s2
        dy_dq3 = self.sin_r * g
        dz_dq3 = -self.ell * s3 * s2 - self.cos_r * g * c2

        return np.array(
            [
                [
                    2.0 * x0 * dx_dq2,
                    2.0 * (x0 * dx_dq3 + y0 * dy_dq3),
                ],
                [
                    dz_dq2,
                    dz_dq3,
                ],
            ],
            dtype=float,
        )

    def recover_q1(self, q23: np.ndarray, target: np.ndarray) -> float:
        """Recover q1 from one reduced root."""
        q2, q3 = np.asarray(q23, dtype=float)
        c = self.constants

        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)

        h = c.a3 + self.ell * s3
        g = self.ell * c3

        x0 = -c.a6 + (c.d5 - g) * c2 - self.cos_r * h * s2
        y0 = self.sin_r * h

        target_azimuth = math.atan2(float(target[1]), float(target[0]))
        reference_azimuth = math.atan2(float(y0), float(x0))

        return float(wrap_to_pi(target_azimuth - reference_azimuth))

    def direct_dh_fk_and_det(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Independent standard-DH calculation used for validation."""
        d, a, alpha = self.dh_parameters
        q = np.asarray(q, dtype=float)

        transform = np.eye(4)
        origins = [np.zeros(3)]
        z_axes = [np.array([0.0, 0.0, 1.0])]

        for theta, di, ai, alphai in zip(q, d, a, alpha):
            ct = math.cos(float(theta))
            st = math.sin(float(theta))
            ca = math.cos(float(alphai))
            sa = math.sin(float(alphai))

            link_transform = np.array(
                [
                    [ct, -st * ca, st * sa, ai * ct],
                    [st, ct * ca, -ct * sa, ai * st],
                    [0.0, sa, ca, di],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

            transform = transform @ link_transform
            origins.append(transform[:3, 3].copy())
            z_axes.append(transform[:3, 2].copy())

        position = origins[-1]
        jacobian = np.column_stack(
            [
                np.cross(z_axes[index], position - origins[index])
                for index in range(3)
            ]
        )

        return position, float(np.linalg.det(jacobian))


# =============================================================================
# Lightweight outward-expanded interval arithmetic
# =============================================================================


def round_down(value: float) -> float:
    return float(np.nextafter(float(value), -np.inf))


def round_up(value: float) -> float:
    return float(np.nextafter(float(value), np.inf))


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"Invalid interval [{self.lo}, {self.hi}].")

    @staticmethod
    def point(value: float) -> "Interval":
        return Interval(float(value), float(value))

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def __add__(self, other) -> "Interval":
        other = as_interval(other)
        return Interval(
            round_down(self.lo + other.lo),
            round_up(self.hi + other.hi),
        )

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(round_down(-self.hi), round_up(-self.lo))

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
    first = math.ceil((lo - phase) / period)
    last = math.floor((hi - phase) / period)
    return first <= last


def interval_sin(interval: Interval) -> Interval:
    if interval.width >= TWO_PI:
        return Interval(-1.0, 1.0)

    values = [math.sin(interval.lo), math.sin(interval.hi)]
    lower = min(values)
    upper = max(values)

    if interval_contains_phase(interval.lo, interval.hi, PI / 2.0, TWO_PI):
        upper = 1.0
    if interval_contains_phase(interval.lo, interval.hi, -PI / 2.0, TWO_PI):
        lower = -1.0

    return Interval(round_down(lower), round_up(upper))


def interval_cos(interval: Interval) -> Interval:
    if interval.width >= TWO_PI:
        return Interval(-1.0, 1.0)

    values = [math.cos(interval.lo), math.cos(interval.hi)]
    lower = min(values)
    upper = max(values)

    if interval_contains_phase(interval.lo, interval.hi, 0.0, TWO_PI):
        upper = 1.0
    if interval_contains_phase(interval.lo, interval.hi, PI, TWO_PI):
        lower = -1.0

    return Interval(round_down(lower), round_up(upper))


def determinant_interval(
    model: EquivalentPanda3R,
    q2_interval: Interval,
    q3_interval: Interval,
) -> Interval:
    """Natural interval extension of the analytic determinant formula."""
    s2 = interval_sin(q2_interval)
    c2 = interval_cos(q2_interval)
    s3 = interval_sin(q3_interval)
    c3 = interval_cos(q3_interval)

    c = model.constants
    h = c.a3 + model.ell * s3

    u = c.a3 * c3 + c.d5 * s3
    v = model.cos_r * h * s2 + (model.ell * c3 - c.d5) * c2
    w = c.a6 * (
        c.a3 * c3 * model.cos_r**2
        + c.d5 * s3
        - model.ell * s3 * c3 * model.sin_r**2
    )

    return model.ell * (u * v + w)


# =============================================================================
# Numerical all-roots search for position IK
# =============================================================================


@dataclass
class IKSolveResult:
    solutions: list[np.ndarray]
    elapsed: float
    attempted_seeds: int


def solve_position_ik_multistart(
    model: EquivalentPanda3R,
    target: np.ndarray,
    starts_per_axis: int,
    residual_tolerance: float = 1e-9,
    duplicate_tolerance: float = 1e-6,
) -> IKSolveResult:
    """
    Numerically collect distinct position-IK roots.

    This is deliberately shared by both compared methods so that the timing
    difference concerns only the connectivity test, not a different IK solver.
    """
    start_time = time.perf_counter()
    target = np.asarray(target, dtype=float)

    seeds = np.linspace(-PI, PI, starts_per_axis, endpoint=False)
    reduced_roots: list[np.ndarray] = []

    for q2_seed in seeds:
        for q3_seed in seeds:
            result = root(
                fun=lambda q23: model.reduced_equations(q23, target),
                x0=np.array([q2_seed, q3_seed], dtype=float),
                jac=model.reduced_jacobian,
                method="hybr",
                options={"xtol": 1e-11, "maxfev": 300},
            )

            if not result.success:
                continue

            candidate = wrap_to_pi(result.x)
            residual = np.linalg.norm(
                model.reduced_equations(candidate, target)
            )
            if residual > residual_tolerance:
                continue

            duplicate = any(
                np.linalg.norm(angular_difference(candidate, existing))
                < duplicate_tolerance
                for existing in reduced_roots
            )
            if not duplicate:
                reduced_roots.append(np.asarray(candidate, dtype=float))

    full_solutions: list[np.ndarray] = []

    for q23 in reduced_roots:
        q1 = model.recover_q1(q23, target)
        q = np.array([q1, q23[0], q23[1]], dtype=float)

        residual = np.linalg.norm(model.fk_position(q) - target)
        if residual <= 1e-7:
            full_solutions.append(q)

    full_solutions.sort(key=lambda q: (q[2], q[1], q[0]))

    return IKSolveResult(
        solutions=full_solutions,
        elapsed=time.perf_counter() - start_time,
        attempted_seeds=starts_per_axis**2,
    )


# =============================================================================
# Method A: same-sign IK pairs and straight-line joint interpolation
# =============================================================================


@dataclass
class LinearWitness:
    first_index: int
    second_index: int
    lifted_goal: np.ndarray
    sampled_min_abs_det: float
    certified: bool
    interval_nodes: int
    paths_tested: int


def certify_linear_segment(
    model: EquivalentPanda3R,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    determinant_margin: float,
    maximum_depth: int,
) -> tuple[bool, int]:
    """
    Certify det(J_v) != 0 on one straight segment by recursively enclosing
    short segment pieces in q2-q3 boxes.

    The interval boxes contain more points than the line itself, so this test
    is conservative: failure means inconclusive, not singular.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    stack = [(0.0, 1.0, 0)]
    visited = 0

    while stack:
        t0, t1, depth = stack.pop()
        visited += 1

        qa = (1.0 - t0) * q_start + t0 * q_goal
        qb = (1.0 - t1) * q_start + t1 * q_goal

        q2_interval = Interval(
            min(float(qa[1]), float(qb[1])),
            max(float(qa[1]), float(qb[1])),
        )
        q3_interval = Interval(
            min(float(qa[2]), float(qb[2])),
            max(float(qa[2]), float(qb[2])),
        )

        determinant_bounds = determinant_interval(
            model,
            q2_interval,
            q3_interval,
        )

        if (
            determinant_bounds.lo >= determinant_margin
            or determinant_bounds.hi <= -determinant_margin
        ):
            continue

        if depth >= maximum_depth:
            return False, visited

        midpoint = 0.5 * (t0 + t1)
        stack.append((t0, midpoint, depth + 1))
        stack.append((midpoint, t1, depth + 1))

    return True, visited


def periodic_goal_lifts(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lift_radius: int,
) -> Iterable[np.ndarray]:
    """
    Generate nearby torus lifts of q_goal.

    q1 does not affect det(J_v), so it uses its shortest lift.  q2 and q3
    examine nearby homotopy classes k in [-lift_radius, lift_radius].
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    shortest_goal = q_start + angular_difference(q_goal, q_start)

    for k2, k3 in product(
        range(-lift_radius, lift_radius + 1),
        repeat=2,
    ):
        lifted = shortest_goal.copy()
        lifted[1] += TWO_PI * k2
        lifted[2] += TWO_PI * k3
        yield lifted


def find_linear_witness(
    model: EquivalentPanda3R,
    solutions: list[np.ndarray],
    path_samples: int,
    lift_radius: int,
    determinant_margin: float,
    certify_path: bool,
    certification_depth: int,
) -> Optional[LinearWitness]:
    """Apply the existing random-target/straight-path test to one target."""
    paths_tested = 0

    for first_index, second_index in combinations(range(len(solutions)), 2):
        q_first = solutions[first_index]
        q_second = solutions[second_index]

        det_first = float(model.det_jv(q_first))
        det_second = float(model.det_jv(q_second))

        if (
            abs(det_first) <= determinant_margin
            or abs(det_second) <= determinant_margin
            or np.sign(det_first) != np.sign(det_second)
        ):
            continue

        for lifted_goal in periodic_goal_lifts(
            q_first,
            q_second,
            lift_radius=lift_radius,
        ):
            paths_tested += 1
            interpolation = np.linspace(0.0, 1.0, path_samples)
            path = (
                (1.0 - interpolation[:, None]) * q_first
                + interpolation[:, None] * lifted_goal
            )
            determinant_values = np.asarray(model.det_jv(path), dtype=float)

            same_sign = np.all(
                np.sign(det_first) * determinant_values > determinant_margin
            )
            if not same_sign:
                continue

            sampled_minimum = float(np.min(np.abs(determinant_values)))

            if certify_path:
                certified, interval_nodes = certify_linear_segment(
                    model=model,
                    q_start=q_first,
                    q_goal=lifted_goal,
                    determinant_margin=determinant_margin,
                    maximum_depth=certification_depth,
                )
                if not certified:
                    continue
            else:
                certified = False
                interval_nodes = 0

            return LinearWitness(
                first_index=first_index,
                second_index=second_index,
                lifted_goal=lifted_goal,
                sampled_min_abs_det=sampled_minimum,
                certified=certified,
                interval_nodes=interval_nodes,
                paths_tested=paths_tested,
            )

    return None


# =============================================================================
# Method B: adaptive certified-box GCS on the q2-q3 torus
# =============================================================================


@dataclass(frozen=True)
class CertifiedBox:
    q2_lo: float
    q2_hi: float
    q3_lo: float
    q3_hi: float
    i0: int
    i1: int
    j0: int
    j1: int
    sign: int
    margin: float
    depth: int


@dataclass
class GCSAtlas:
    model: EquivalentPanda3R
    maximum_depth: int
    determinant_margin: float
    boxes: list[CertifiedBox]
    owner: np.ndarray
    adjacency: list[set[int]]
    component_labels: np.ndarray
    component_count_by_sign: dict[int, int]
    unresolved_cells: int
    certified_fraction: float
    build_time: float
    boxes_tested: int
    edge_count: int

    def locate(self, q23: np.ndarray) -> int:
        """Locate one wrapped q2-q3 point in the fine owner grid."""
        q2, q3 = wrap_to_pi(q23)
        size = self.owner.shape[0]

        i = int(math.floor((float(q2) + PI) / TWO_PI * size)) % size
        j = int(math.floor((float(q3) + PI) / TWO_PI * size)) % size

        region = int(self.owner[i, j])
        if region >= 0:
            return region

        # A root exactly on a fine-cell boundary can be assigned to a nearby
        # certified cell due to floating-point wrapping.  Search a small ring.
        for radius in (1, 2):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    candidate = int(
                        self.owner[(i + di) % size, (j + dj) % size]
                    )
                    if candidate < 0:
                        continue
                    box = self.boxes[candidate]

                    # Compare after trying nearby 2pi lifts.
                    for shift2, shift3 in product((-1, 0, 1), repeat=2):
                        q2_lift = float(q2) + shift2 * TWO_PI
                        q3_lift = float(q3) + shift3 * TWO_PI
                        if (
                            box.q2_lo - 1e-12 <= q2_lift <= box.q2_hi + 1e-12
                            and box.q3_lo - 1e-12
                            <= q3_lift
                            <= box.q3_hi + 1e-12
                        ):
                            return candidate

        return -1

    def graph_path(self, start: int, goal: int) -> list[int]:
        """Breadth-first region path, used only to report a witness."""
        if start == goal:
            return [start]

        predecessor = {start: -1}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbour in self.adjacency[current]:
                if neighbour in predecessor:
                    continue
                predecessor[neighbour] = current
                if neighbour == goal:
                    queue.clear()
                    break
                queue.append(neighbour)

        if goal not in predecessor:
            return []

        path = [goal]
        current = goal
        while predecessor[current] != -1:
            current = predecessor[current]
            path.append(current)
        path.reverse()
        return path


@dataclass
class GCSWitness:
    first_index: int
    second_index: int
    first_region: int
    second_region: int
    component: int
    region_path: list[int]
    located_solutions: int


def build_gcs_atlas(
    model: EquivalentPanda3R,
    maximum_depth: int,
    determinant_margin: float,
) -> GCSAtlas:
    """Build the adaptive certified convex-box graph for one theta_r slice."""
    start_time = time.perf_counter()
    grid_size = 2**maximum_depth
    owner = np.full((grid_size, grid_size), -1, dtype=np.int32)

    boxes: list[CertifiedBox] = []
    boxes_tested = 0
    unresolved_cells = 0

    # Integer bounds refer to the finest grid and make adjacency exact.
    stack = [(0, grid_size, 0, grid_size, 0)]

    def coordinate(index: int) -> float:
        return -PI + TWO_PI * index / grid_size

    while stack:
        i0, i1, j0, j1, depth = stack.pop()
        boxes_tested += 1

        q2_lo = coordinate(i0)
        q2_hi = coordinate(i1)
        q3_lo = coordinate(j0)
        q3_hi = coordinate(j1)

        determinant_bounds = determinant_interval(
            model,
            Interval(q2_lo, q2_hi),
            Interval(q3_lo, q3_hi),
        )

        if determinant_bounds.lo >= determinant_margin:
            sign = +1
            margin = determinant_bounds.lo
        elif determinant_bounds.hi <= -determinant_margin:
            sign = -1
            margin = -determinant_bounds.hi
        else:
            sign = 0
            margin = 0.0

        if sign != 0:
            region = len(boxes)
            boxes.append(
                CertifiedBox(
                    q2_lo=q2_lo,
                    q2_hi=q2_hi,
                    q3_lo=q3_lo,
                    q3_hi=q3_hi,
                    i0=i0,
                    i1=i1,
                    j0=j0,
                    j1=j1,
                    sign=sign,
                    margin=float(margin),
                    depth=depth,
                )
            )
            owner[i0:i1, j0:j1] = region
            continue

        if depth >= maximum_depth:
            unresolved_cells += (i1 - i0) * (j1 - j0)
            continue

        i_mid = (i0 + i1) // 2
        j_mid = (j0 + j1) // 2
        next_depth = depth + 1

        stack.extend(
            [
                (i0, i_mid, j0, j_mid, next_depth),
                (i_mid, i1, j0, j_mid, next_depth),
                (i0, i_mid, j_mid, j1, next_depth),
                (i_mid, i1, j_mid, j1, next_depth),
            ]
        )

    adjacency: list[set[int]] = [set() for _ in boxes]
    edge_pairs: set[tuple[int, int]] = set()

    # Edge and corner contacts are valid because every accepted box is
    # certified on its closed boundary.  The diagonal checks also avoid
    # artificial fragmentation at adaptive-grid vertices.  np.roll includes
    # the periodic identifications q2=-pi~pi and q3=-pi~pi.
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        neighbour_owner = np.roll(owner, shift=(-di, -dj), axis=(0, 1))
        mask = (
            (owner >= 0)
            & (neighbour_owner >= 0)
            & (owner != neighbour_owner)
        )

        first = owner[mask]
        second = neighbour_owner[mask]

        for region_a, region_b in zip(first.tolist(), second.tolist()):
            a = int(region_a)
            b = int(region_b)
            if boxes[a].sign != boxes[b].sign:
                continue
            if a > b:
                a, b = b, a
            edge_pairs.add((a, b))

    for a, b in edge_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    component_labels = np.full(len(boxes), -1, dtype=np.int32)
    component_signs: list[int] = []

    for start in range(len(boxes)):
        if component_labels[start] >= 0:
            continue

        label = len(component_signs)
        component_signs.append(boxes[start].sign)
        component_labels[start] = label
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if component_labels[neighbour] >= 0:
                    continue
                component_labels[neighbour] = label
                queue.append(neighbour)

    component_count_by_sign = {
        -1: sum(sign < 0 for sign in component_signs),
        +1: sum(sign > 0 for sign in component_signs),
    }

    return GCSAtlas(
        model=model,
        maximum_depth=maximum_depth,
        determinant_margin=determinant_margin,
        boxes=boxes,
        owner=owner,
        adjacency=adjacency,
        component_labels=component_labels,
        component_count_by_sign=component_count_by_sign,
        unresolved_cells=unresolved_cells,
        certified_fraction=float(np.mean(owner >= 0)),
        build_time=time.perf_counter() - start_time,
        boxes_tested=boxes_tested,
        edge_count=len(edge_pairs),
    )


def find_gcs_witness(
    atlas: GCSAtlas,
    solutions: list[np.ndarray],
) -> Optional[GCSWitness]:
    """Find two distinct IK roots in one certified graph component."""
    regions = [atlas.locate(q[1:]) for q in solutions]
    components = [
        int(atlas.component_labels[region]) if region >= 0 else -1
        for region in regions
    ]

    candidates = []
    for first, second in combinations(range(len(solutions)), 2):
        if regions[first] < 0 or regions[second] < 0:
            continue
        if components[first] != components[second]:
            continue

        separation = float(
            np.linalg.norm(angular_difference(solutions[first], solutions[second]))
        )
        candidates.append((separation, first, second))

    if not candidates:
        return None

    # Prefer a well-separated pair, avoiding duplicate-root numerical noise.
    candidates.sort(reverse=True)
    _, first, second = candidates[0]

    region_path = atlas.graph_path(regions[first], regions[second])
    if not region_path:
        return None

    return GCSWitness(
        first_index=first,
        second_index=second,
        first_region=regions[first],
        second_region=regions[second],
        component=components[first],
        region_path=region_path,
        located_solutions=sum(region >= 0 for region in regions),
    )


# =============================================================================
# Fair per-slice comparison
# =============================================================================


@dataclass
class MethodTiming:
    found: bool = False
    target_index: int = -1
    ik_time: float = 0.0
    connectivity_time: float = 0.0
    witness_pair: tuple[int, int] | None = None
    witness_target: np.ndarray | None = None
    witness_solutions: list[np.ndarray] | None = None
    detail: object | None = None


@dataclass
class SliceComparison:
    theta_r: float
    validation_position_error: float
    validation_det_error: float
    atlas: GCSAtlas
    linear: MethodTiming
    gcs: MethodTiming
    targets_solved: int
    total_ik_calls: int


def validate_equivalent_model(
    model: EquivalentPanda3R,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    maximum_position_error = 0.0
    maximum_det_error = 0.0
    rtb_equivalent = model.build_rtb_robot()

    for _ in range(samples):
        q = rng.uniform(-PI, PI, size=3)

        if rtb_equivalent is not None:
            reference_position = np.asarray(
                rtb_equivalent.fkine(q).t,
                dtype=float,
            ).reshape(3)
            reference_det = float(
                np.linalg.det(np.asarray(rtb_equivalent.jacob0(q))[:3, :])
            )
        else:
            reference_position, reference_det = model.direct_dh_fk_and_det(q)

        closed_position = model.fk_position(q)
        closed_det = float(model.det_jv(q))

        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(reference_position - closed_position)),
        )
        maximum_det_error = max(
            maximum_det_error,
            abs(reference_det - closed_det),
        )

    return maximum_position_error, maximum_det_error


def generate_reachable_targets(
    model: EquivalentPanda3R,
    count: int,
    seed: int,
    minimum_abs_det: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (generating configuration, target position) pairs."""
    rng = np.random.default_rng(seed)
    targets: list[tuple[np.ndarray, np.ndarray]] = []

    maximum_attempts = 100 * count
    attempts = 0

    while len(targets) < count and attempts < maximum_attempts:
        attempts += 1
        q = rng.uniform(-PI, PI, size=3)

        if abs(float(model.det_jv(q))) < minimum_abs_det:
            continue

        target = np.asarray(model.fk_position(q), dtype=float)
        if math.hypot(float(target[0]), float(target[1])) < 1e-5:
            continue

        targets.append((q, target))

    if len(targets) < count:
        raise RuntimeError(
            f"Could generate only {len(targets)} regular targets out of {count}."
        )

    return targets


def compare_one_slice(
    theta_r: float,
    constants: PandaPaperConstants,
    args: argparse.Namespace,
    slice_index: int,
) -> SliceComparison:
    model = EquivalentPanda3R(theta_r=theta_r, constants=constants)

    validation_position_error, validation_det_error = validate_equivalent_model(
        model,
        samples=args.validation_samples,
        seed=args.seed + 17 * slice_index,
    )

    atlas = build_gcs_atlas(
        model=model,
        maximum_depth=args.gcs_depth,
        determinant_margin=args.det_margin,
    )

    targets = generate_reachable_targets(
        model=model,
        count=args.max_targets,
        seed=args.seed + 1009 * slice_index,
        minimum_abs_det=args.target_det_margin,
    )

    linear_timing = MethodTiming()
    gcs_timing = MethodTiming()
    total_ik_calls = 0

    for target_index, (_, target) in enumerate(targets, start=1):
        ik_result = solve_position_ik_multistart(
            model=model,
            target=target,
            starts_per_axis=args.ik_starts,
        )
        total_ik_calls += ik_result.attempted_seeds

        # Count the shared IK cost toward each method only while that method is
        # still searching. This gives a fair end-to-end time-to-first-witness.
        if not linear_timing.found:
            linear_timing.ik_time += ik_result.elapsed
        if not gcs_timing.found:
            gcs_timing.ik_time += ik_result.elapsed

        solutions = ik_result.solutions

        if not linear_timing.found:
            start = time.perf_counter()
            linear_witness = find_linear_witness(
                model=model,
                solutions=solutions,
                path_samples=args.path_samples,
                lift_radius=args.lift_radius,
                determinant_margin=args.det_margin,
                certify_path=not args.no_linear_certificate,
                certification_depth=args.linear_cert_depth,
            )
            linear_timing.connectivity_time += time.perf_counter() - start

            if linear_witness is not None:
                linear_timing.found = True
                linear_timing.target_index = target_index
                linear_timing.witness_pair = (
                    linear_witness.first_index,
                    linear_witness.second_index,
                )
                linear_timing.witness_target = target.copy()
                linear_timing.witness_solutions = [q.copy() for q in solutions]
                linear_timing.detail = linear_witness

        if not gcs_timing.found:
            start = time.perf_counter()
            gcs_witness = find_gcs_witness(atlas, solutions)
            gcs_timing.connectivity_time += time.perf_counter() - start

            if gcs_witness is not None:
                gcs_timing.found = True
                gcs_timing.target_index = target_index
                gcs_timing.witness_pair = (
                    gcs_witness.first_index,
                    gcs_witness.second_index,
                )
                gcs_timing.witness_target = target.copy()
                gcs_timing.witness_solutions = [q.copy() for q in solutions]
                gcs_timing.detail = gcs_witness

        if linear_timing.found and gcs_timing.found:
            return SliceComparison(
                theta_r=theta_r,
                validation_position_error=validation_position_error,
                validation_det_error=validation_det_error,
                atlas=atlas,
                linear=linear_timing,
                gcs=gcs_timing,
                targets_solved=target_index,
                total_ik_calls=total_ik_calls,
            )

    return SliceComparison(
        theta_r=theta_r,
        validation_position_error=validation_position_error,
        validation_det_error=validation_det_error,
        atlas=atlas,
        linear=linear_timing,
        gcs=gcs_timing,
        targets_solved=args.max_targets,
        total_ik_calls=total_ik_calls,
    )


# =============================================================================
# Reporting and plots
# =============================================================================


def method_total_time(method: MethodTiming) -> float:
    return method.ik_time + method.connectivity_time


def print_slice_report(result: SliceComparison) -> None:
    theta = result.theta_r
    atlas = result.atlas

    print("\n" + "=" * 88)
    print(
        f"theta_r = {theta:.10f} rad = {math.degrees(theta):.3f} deg"
    )
    print("=" * 88)
    print(
        "Equivalent-chain validation: "
        f"position error={result.validation_position_error:.3e}, "
        f"det error={result.validation_det_error:.3e}"
    )
    print(
        "GCS atlas: "
        f"build={atlas.build_time:.4f} s, "
        f"boxes={len(atlas.boxes)}, edges={atlas.edge_count}, "
        f"coverage={100.0 * atlas.certified_fraction:.3f}%, "
        f"components(-/+)="
        f"{atlas.component_count_by_sign[-1]}/"
        f"{atlas.component_count_by_sign[+1]}"
    )

    linear = result.linear
    print("\nMethod A - random target + straight path")
    print(f"  witness found       : {linear.found}")
    print(f"  target index        : {linear.target_index}")
    print(f"  cumulative IK time  : {linear.ik_time:.4f} s")
    print(f"  path-test time      : {linear.connectivity_time:.4f} s")
    print(f"  online total        : {method_total_time(linear):.4f} s")
    if isinstance(linear.detail, LinearWitness):
        print(f"  IK pair             : {linear.witness_pair}")
        print(
            "  sampled min |det|   : "
            f"{linear.detail.sampled_min_abs_det:.6e}"
        )
        print(f"  interval certified  : {linear.detail.certified}")
        print(f"  paths tested        : {linear.detail.paths_tested}")

    gcs = result.gcs
    print("\nMethod B - certified-box GCS")
    print(f"  witness found       : {gcs.found}")
    print(f"  target index        : {gcs.target_index}")
    print(f"  cumulative IK time  : {gcs.ik_time:.4f} s")
    print(f"  lookup time         : {gcs.connectivity_time:.6f} s")
    print(f"  online total        : {method_total_time(gcs):.4f} s")
    print(
        "  first-use total     : "
        f"{atlas.build_time + method_total_time(gcs):.4f} s"
    )
    if isinstance(gcs.detail, GCSWitness):
        print(f"  IK pair             : {gcs.witness_pair}")
        print(f"  graph component     : {gcs.detail.component}")
        print(f"  region-path length  : {len(gcs.detail.region_path)}")
        print(
            "  located IK roots    : "
            f"{gcs.detail.located_solutions}/"
            f"{len(gcs.witness_solutions or [])}"
        )


def write_csv(results: list[SliceComparison], filename: Path) -> None:
    fields = [
        "theta_r_rad",
        "theta_r_deg",
        "linear_found",
        "linear_target_index",
        "linear_ik_time_s",
        "linear_connectivity_time_s",
        "linear_total_s",
        "gcs_found",
        "gcs_target_index",
        "gcs_build_time_s",
        "gcs_ik_time_s",
        "gcs_lookup_time_s",
        "gcs_online_total_s",
        "gcs_first_use_total_s",
        "gcs_boxes",
        "gcs_edges",
        "gcs_certified_fraction",
        "negative_components",
        "positive_components",
        "validation_position_error",
        "validation_det_error",
    ]

    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for result in results:
            atlas = result.atlas
            writer.writerow(
                {
                    "theta_r_rad": result.theta_r,
                    "theta_r_deg": math.degrees(result.theta_r),
                    "linear_found": result.linear.found,
                    "linear_target_index": result.linear.target_index,
                    "linear_ik_time_s": result.linear.ik_time,
                    "linear_connectivity_time_s": result.linear.connectivity_time,
                    "linear_total_s": method_total_time(result.linear),
                    "gcs_found": result.gcs.found,
                    "gcs_target_index": result.gcs.target_index,
                    "gcs_build_time_s": atlas.build_time,
                    "gcs_ik_time_s": result.gcs.ik_time,
                    "gcs_lookup_time_s": result.gcs.connectivity_time,
                    "gcs_online_total_s": method_total_time(result.gcs),
                    "gcs_first_use_total_s": (
                        atlas.build_time + method_total_time(result.gcs)
                    ),
                    "gcs_boxes": len(atlas.boxes),
                    "gcs_edges": atlas.edge_count,
                    "gcs_certified_fraction": atlas.certified_fraction,
                    "negative_components": atlas.component_count_by_sign[-1],
                    "positive_components": atlas.component_count_by_sign[+1],
                    "validation_position_error": result.validation_position_error,
                    "validation_det_error": result.validation_det_error,
                }
            )


def make_summary_plot(results: list[SliceComparison], filename: Path) -> None:
    labels = [
        rf"$\pi/{round(PI / result.theta_r):d}$"
        if abs(PI / result.theta_r - round(PI / result.theta_r)) < 1e-8
        else f"{math.degrees(result.theta_r):.1f} deg"
        for result in results
    ]
    x = np.arange(len(results), dtype=float)
    width = 0.25

    linear_times = np.array(
        [method_total_time(result.linear) for result in results],
        dtype=float,
    )
    gcs_online = np.array(
        [method_total_time(result.gcs) for result in results],
        dtype=float,
    )
    gcs_first = np.array(
        [
            result.atlas.build_time + method_total_time(result.gcs)
            for result in results
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    axis.bar(x - width, linear_times, width, label="Linear-path method")
    axis.bar(x, gcs_online, width, label="GCS online only")
    axis.bar(x + width, gcs_first, width, label="GCS first use (offline + online)")

    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_xlabel(r"Redundant angle $\theta_r$")
    axis.set_ylabel("Time to first witness [s]")
    axis.set_title("Panda parameterised-3R cuspidality-method comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.savefig(filename, dpi=180)
    plt.close(figure)


def parse_theta_values(text: str) -> list[float]:
    """
    Parse comma-separated values. Accepted tokens include pi/12, pi/6,
    -pi/3, 0.5, and 30deg.
    """
    values = []

    for raw_token in text.split(","):
        token = raw_token.strip().lower().replace(" ", "")
        if not token:
            continue

        if token.endswith("deg"):
            values.append(math.radians(float(token[:-3])))
            continue

        sign = 1.0
        if token.startswith("-"):
            sign = -1.0
            token = token[1:]

        if token == "pi":
            values.append(sign * PI)
        elif token.startswith("pi/"):
            values.append(sign * PI / float(token[3:]))
        elif token.startswith("pi*"):
            values.append(sign * PI * float(token[3:]))
        else:
            values.append(sign * float(token))

    if not values:
        raise ValueError("No theta_r values were parsed.")

    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare straight-line IK-pair search and a certified-box GCS "
            "on parameterised Panda 3R slices."
        )
    )

    parser.add_argument(
        "--theta-values",
        default="pi/12,pi/6,pi/3,pi/2",
        help="Comma-separated theta_r slices (default: paper's Fig. 8 slices).",
    )
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--ik-starts", type=int, default=17)
    parser.add_argument("--path-samples", type=int, default=1001)
    parser.add_argument("--lift-radius", type=int, default=1)
    parser.add_argument("--linear-cert-depth", type=int, default=16)
    parser.add_argument("--no-linear-certificate", action="store_true")
    parser.add_argument("--gcs-depth", type=int, default=10)
    parser.add_argument("--det-margin", type=float, default=1e-7)
    parser.add_argument("--target-det-margin", type=float, default=1e-4)
    parser.add_argument("--validation-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output-prefix",
        default="panda_cuspidality_comparison",
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    theta_values = parse_theta_values(args.theta_values)

    robot, constants = load_panda_and_constants()

    print("=" * 88)
    print("Panda cuspidality comparison on the paper's parameterised 3R reduction")
    print("=" * 88)
    if robot is None:
        print(
            "WARNING: roboticstoolbox-python is not installed in this runtime; "
            "using the published constants. On a machine with RTB installed, "
            "the script executes robot = rtb.models.DH.Panda() and verifies "
            "the extracted constants."
        )
    else:
        print(f"Robot: {robot.name}, n={robot.n}")

    print(f"Parameter source: {constants.source}")
    print(
        "Constants [m]: "
        f"d3={constants.d3}, a3={constants.a3}, d5={constants.d5}, "
        f"a6={constants.a6}, d7={constants.d7}"
    )
    print(
        "Important: the paper's Table-V mapping excludes theta_r=0,+/-pi. "
        "This script compares fixed regular slices; it is not by itself a "
        "proof of the paper's 'almost all theta_r' condition."
    )

    results: list[SliceComparison] = []
    total_start = time.perf_counter()

    for slice_index, theta_r in enumerate(theta_values):
        result = compare_one_slice(
            theta_r=theta_r,
            constants=constants,
            args=args,
            slice_index=slice_index,
        )
        results.append(result)
        print_slice_report(result)

    output_prefix = Path(args.output_prefix)
    csv_file = output_prefix.with_suffix(".csv")
    plot_file = output_prefix.with_suffix(".png")

    write_csv(results, csv_file)
    make_summary_plot(results, plot_file)

    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    print(
        "theta_r    linear target / total     GCS target / online     "
        "GCS build / first-use"
    )
    for result in results:
        print(
            f"{result.theta_r:8.5f}    "
            f"{result.linear.target_index:3d} / "
            f"{method_total_time(result.linear):8.4f}s      "
            f"{result.gcs.target_index:3d} / "
            f"{method_total_time(result.gcs):8.4f}s      "
            f"{result.atlas.build_time:8.4f}s / "
            f"{result.atlas.build_time + method_total_time(result.gcs):8.4f}s"
        )

    print(f"\nCSV:  {csv_file}")
    print(f"Plot: {plot_file}")
    print(f"Wall-clock runtime: {time.perf_counter() - total_start:.3f} s")


if __name__ == "__main__":
    main()