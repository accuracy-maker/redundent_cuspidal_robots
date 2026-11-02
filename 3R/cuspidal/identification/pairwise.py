"""
Random-target, IK-pair, straight-line cuspidality test for the supplied
3R standard-DH robot.

Procedure:
    1. Sample a random joint configuration and compute a reachable position.
    2. Compute all numerically discovered position-IK solutions.
    3. Select IK pairs having the same nonzero sign of det(J_v).
    4. Test direct linear interpolation between each pair.

A successful path is a sufficient cuspidality witness.

Important:
    Failure to find a straight-line path does not prove that the robot is
    noncuspidal, because a curved nonsingular path may still exist.
"""

from itertools import combinations
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

import roboticstoolbox as rtb



# =============================================================================
# Robot definition
# =============================================================================

if rtb is not None:
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
else:
    robot = None


PI = math.pi
TWO_PI = 2.0 * PI


def wrap_to_pi(q):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(q, dtype=float) + PI) % TWO_PI - PI


def angle_difference(a, b):
    """Wrapped angular difference a-b."""
    return wrap_to_pi(np.asarray(a) - np.asarray(b))


# =============================================================================
# Closed-form forward kinematics and Jacobian determinant
# =============================================================================

def fk_position(q):
    """
    End-effector position for the supplied standard-DH robot.
    """
    q1, q2, q3 = np.asarray(q, dtype=float)

    length = 2.0 + 1.5 * np.cos(q3)
    radial_a = 1.0 + length * np.cos(q2)
    radial_b = 1.0 + 1.5 * np.sin(q3)

    return np.array(
        [
            np.cos(q1) * radial_a - np.sin(q1) * radial_b,
            np.sin(q1) * radial_a + np.cos(q1) * radial_b,
            -length * np.sin(q2),
        ],
        dtype=float,
    )


def det_jv(q):
    """
    Determinant of the 3x3 translational Jacobian

        J_v = robot.jacob0(q)[:3, :].

    For this robot, det(J_v) is independent of q1.

    The function is vectorised over arrays whose final dimension is 3.
    """
    q = np.asarray(q, dtype=float)

    q2 = q[..., 1]
    q3 = q[..., 2]

    length = 2.0 + 1.5 * np.cos(q3)

    singular_factor = (
        np.sin(q3)
        + np.cos(q2)
        * (2.0 * np.sin(q3) - np.cos(q3))
    )

    return 1.5 * length * singular_factor


# =============================================================================
# Position IK
# =============================================================================

def reduced_ik(q23, target):
    """
    Reduced position-IK equations in q2 and q3.

    q1 is eliminated by using the squared radial distance x^2+y^2.
    """
    q2, q3 = np.asarray(q23, dtype=float)
    x, y, z = np.asarray(target, dtype=float)

    length = 2.0 + 1.5 * np.cos(q3)
    radial_a = 1.0 + length * np.cos(q2)
    radial_b = 1.0 + 1.5 * np.sin(q3)

    return np.array(
        [
            radial_a**2 + radial_b**2 - (x**2 + y**2),
            -length * np.sin(q2) - z,
        ],
        dtype=float,
    )


def reduced_ik_jacobian(q23):
    """
    Analytic Jacobian of reduced_ik().
    """
    q2, q3 = np.asarray(q23, dtype=float)

    s2 = np.sin(q2)
    c2 = np.cos(q2)
    s3 = np.sin(q3)
    c3 = np.cos(q3)

    length = 2.0 + 1.5 * c3

    radial_a = 1.0 + length * c2
    radial_b = 1.0 + 1.5 * s3

    d_a_d_q2 = -length * s2
    d_a_d_q3 = -1.5 * s3 * c2
    d_b_d_q3 = 1.5 * c3

    return np.array(
        [
            [
                2.0 * radial_a * d_a_d_q2,
                2.0 * radial_a * d_a_d_q3
                + 2.0 * radial_b * d_b_d_q3,
            ],
            [
                -length * c2,
                1.5 * s3 * s2,
            ],
        ],
        dtype=float,
    )


def recover_q1(q2, q3, target):
    """
    Recover q1 after solving the reduced IK problem.
    """
    x, y, _ = np.asarray(target, dtype=float)

    length = 2.0 + 1.5 * np.cos(q3)
    radial_a = 1.0 + length * np.cos(q2)
    radial_b = 1.0 + 1.5 * np.sin(q3)

    q1 = (
        np.arctan2(y, x)
        - np.arctan2(radial_b, radial_a)
    )

    return float(wrap_to_pi(q1))


def solve_all_position_ik(
    target,
    starts_per_axis=21,
    residual_tolerance=1e-9,
    duplicate_tolerance=1e-6,
):
    """
    Numerically collect all distinct position-IK solutions.

    A grid of initial guesses is used for the reduced two-dimensional
    root problem.

    This is a numerical all-roots search, not a symbolic completeness proof.
    """
    roots_23 = []

    seeds = np.linspace(
        -PI,
        PI,
        starts_per_axis,
        endpoint=False,
    )

    for q2_seed in seeds:
        for q3_seed in seeds:
            result = root(
                fun=lambda q23: reduced_ik(q23, target),
                x0=np.array(
                    [q2_seed, q3_seed],
                    dtype=float,
                ),
                jac=reduced_ik_jacobian,
                method="hybr",
            )

            if not result.success:
                continue

            candidate = wrap_to_pi(result.x)

            residual = np.linalg.norm(
                reduced_ik(candidate, target)
            )

            if residual > residual_tolerance:
                continue

            is_duplicate = any(
                np.linalg.norm(
                    angle_difference(candidate, existing)
                )
                < duplicate_tolerance
                for existing in roots_23
            )

            if not is_duplicate:
                roots_23.append(candidate)

    solutions = []

    for q2, q3 in roots_23:
        q1 = recover_q1(q2, q3, target)

        q = np.array(
            [q1, q2, q3],
            dtype=float,
        )

        fk_residual = np.linalg.norm(
            fk_position(q) - target
        )

        if fk_residual < 1e-7:
            solutions.append(q)

    # Deterministic ordering for reproducible output.
    solutions.sort(
        key=lambda q: (
            q[2],
            q[1],
            q[0],
        )
    )

    return solutions


# =============================================================================
# Straight-line path test
# =============================================================================

def determinant_derivative_bound(delta_q):
    """
    Conservative global bound on

        |d det(J_v(q(t))) / dt|

    for a straight path q(t).

    Since det(J_v) is independent of q1, only q2 and q3 contribute.
    """
    delta_q2 = abs(float(delta_q[1]))
    delta_q3 = abs(float(delta_q[2]))

    bound_partial_q2 = (
        1.5 * 3.5 * math.sqrt(5.0)
    )

    bound_partial_q3 = 1.5 * (
        1.5 * 4.0
        + 3.5 * (1.0 + math.sqrt(5.0))
    )

    return (
        bound_partial_q2 * delta_q2
        + bound_partial_q3 * delta_q3
    )


def check_linear_path(
    q_start,
    q_goal,
    number_of_samples=4001,
):
    """
    Test the direct joint-space interpolation

        q(t) = (1-t) q_start + t q_goal.

    No angle wrapping or 2*pi lifting is used here, matching the stated
    algorithm directly.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    t = np.linspace(
        0.0,
        1.0,
        number_of_samples,
    )

    path = (
        (1.0 - t[:, None]) * q_start
        + t[:, None] * q_goal
    )

    determinant_values = np.asarray(
        det_jv(path),
        dtype=float,
    )

    determinant_sign = int(
        np.sign(determinant_values[0])
    )

    same_sign = (
        determinant_sign != 0
        and np.all(
            determinant_sign * determinant_values > 0.0
        )
    )

    minimum_sampled_abs_det = float(
        np.min(np.abs(determinant_values))
    )

    time_step = 1.0 / (number_of_samples - 1)

    derivative_bound = determinant_derivative_bound(
        q_goal - q_start
    )

    # Every point is at most time_step/2 from a sampled point.
    conservative_lower_bound = (
        minimum_sampled_abs_det
        - 0.5 * derivative_bound * time_step
    )

    success = bool(
        same_sign
        and conservative_lower_bound > 1e-8
    )

    return {
        "success": success,
        "t": t,
        "path": path,
        "determinants": determinant_values,
        "sign": determinant_sign,
        "minimum_sampled_abs_det":
            minimum_sampled_abs_det,
        "conservative_lower_bound":
            float(conservative_lower_bound),
    }


# =============================================================================
# Four-step cuspidality search
# =============================================================================

def search_for_cuspidality_witness(
    maximum_targets=200,
    random_seed=11,
    starts_per_axis=21,
    path_samples=4001,
):
    rng = np.random.default_rng(random_seed)

    for trial in range(1, maximum_targets + 1):

        # ------------------------------------------------------------------
        # Step 1: generate a reachable random target.
        # ------------------------------------------------------------------

        sampled_q = rng.uniform(
            -PI,
            PI,
            size=3,
        )

        if robot is not None:
            target = np.asarray(
                robot.fkine(sampled_q).t,
                dtype=float,
            ).reshape(3)
        else:
            target = fk_position(sampled_q)

        # Avoid targets generated from nearly singular configurations.
        if abs(float(det_jv(sampled_q))) < 1e-4:
            continue

        # ------------------------------------------------------------------
        # Step 2: compute all numerical IK solutions.
        # ------------------------------------------------------------------

        ik_solutions = solve_all_position_ik(
            target,
            starts_per_axis=starts_per_axis,
        )

        print(
            f"Trial {trial}: "
            f"{len(ik_solutions)} IK solutions"
        )

        determinants = [
            float(det_jv(q))
            for q in ik_solutions
        ]

        # ------------------------------------------------------------------
        # Step 3: select pairs having the same determinant sign.
        # ------------------------------------------------------------------

        for first, second in combinations(
            range(len(ik_solutions)),
            2,
        ):
            det_first = determinants[first]
            det_second = determinants[second]

            if (
                abs(det_first) < 1e-7
                or abs(det_second) < 1e-7
            ):
                continue

            if np.sign(det_first) != np.sign(det_second):
                continue

            # --------------------------------------------------------------
            # Step 4: test direct linear interpolation.
            # --------------------------------------------------------------

            path_result = check_linear_path(
                ik_solutions[first],
                ik_solutions[second],
                number_of_samples=path_samples,
            )

            if path_result["success"]:
                return {
                    "trial": trial,
                    "sampled_q": sampled_q,
                    "target": target,
                    "ik_solutions": ik_solutions,
                    "pair": (first, second),
                    "path_result": path_result,
                }

    return None


# =============================================================================
# Reporting and plotting
# =============================================================================

def report_result(result):
    first, second = result["pair"]
    path_result = result["path_result"]

    print("\n" + "=" * 78)
    print("CUSPIDALITY WITNESS FOUND")
    print("=" * 78)

    print(f"Target trial: {result['trial']}")
    print(f"Random sampled q: {result['sampled_q']}")
    print(f"Target position: {result['target']}")
    print(
        "Number of IK solutions: "
        f"{len(result['ik_solutions'])}"
    )

    for index, q in enumerate(
        result["ik_solutions"]
    ):
        fk_residual = np.linalg.norm(
            fk_position(q) - result["target"]
        )

        print(
            f"IK {index}: "
            f"q={np.array2string(q, precision=10)}, "
            f"det(J_v)={float(det_jv(q)): .10f}, "
            f"FK residual={fk_residual:.3e}"
        )

    print(
        f"Selected same-sign pair: "
        f"IK {first} and IK {second}"
    )

    print(
        "Minimum sampled |det(J_v)|: "
        f"{path_result['minimum_sampled_abs_det']:.10e}"
    )

    print(
        "Conservative lower bound: "
        f"{path_result['conservative_lower_bound']:.10e}"
    )

    print(
        "Conclusion: the direct linear path is nonsingular."
    )

    print(
        "Therefore, the robot is cuspidal for its "
        "3D position kinematics."
    )


def plot_result(
    result,
    output_file="linear_cuspidality_witness.png",
):
    first, second = result["pair"]

    t = result["path_result"]["t"]
    path = result["path_result"]["path"]
    determinant_values = (
        result["path_result"]["determinants"]
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )

    for joint in range(3):
        axes[0].plot(
            t,
            path[:, joint],
            label=fr"$q_{joint + 1}$",
        )

    axes[0].set_xlabel(r"Path parameter $t$")
    axes[0].set_ylabel("Joint angle [rad]")
    axes[0].set_title(
        f"Linear path: IK {first} to IK {second}"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        t,
        determinant_values,
    )

    axes[1].axhline(
        0.0,
        linewidth=1.0,
    )

    axes[1].set_xlabel(r"Path parameter $t$")
    axes[1].set_ylabel(r"$\det J_v$")
    axes[1].set_title(
        "Jacobian determinant along the path"
    )
    axes[1].grid(alpha=0.3)

    figure.savefig(
        output_file,
        dpi=180,
    )

    plt.close(figure)

    print(f"Plot saved to: {output_file}")


def main():
    start_time = time.perf_counter()

    result = search_for_cuspidality_witness(
        maximum_targets=200,
        random_seed=11,
        starts_per_axis=21,
        path_samples=4001,
    )

    if result is None:
        print("\nNo straight-line witness was found.")
        print(
            "This does not prove that the robot is "
            "noncuspidal."
        )
        return

    report_result(result)
    plot_result(result)

    print(
        f"Runtime: "
        f"{time.perf_counter() - start_time:.3f} s"
    )


if __name__ == "__main__":
    main()