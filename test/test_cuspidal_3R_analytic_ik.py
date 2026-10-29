"""
Compare resultant-based IK with multi-start Jacobian IK for the 3R robot.

Target position: [2.5, 0.0, 0.5]

The symbolic resultant is formed with exact rational coefficients. Its real
roots are back-substituted to recover q2 and q3, then q1 is recovered from
the horizontal position equations. Both solution sets are checked using
Robotics Toolbox forward kinematics.
"""

import numpy as np
import sympy as sp
from sympy import init_printing
import roboticstoolbox as rtb
from scipy.optimize import root

init_printing(use_unicode=True)

# NumPy target for numerical calculations.
TARGET = np.array([2.5, 0.0, 0.5], dtype=float)

# Exact SymPy target for polynomial elimination.
X_TARGET = sp.Rational(5, 2)
Y_TARGET = sp.Rational(0)
Z_TARGET = sp.Rational(1, 2)


robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=0.0, a=1.0, alpha=-np.pi / 2),
        rtb.RevoluteDH(d=1.0, a=2.0, alpha=np.pi / 2),
        rtb.RevoluteDH(d=0.0, a=1.5, alpha=0.0),
    ],
    name="3R_cuspidal_robot",
)


def position_formula(q):
    """End-effector position derived from the robot's standard DH model."""
    q1, q2, q3 = q

    K = 2.0 + 1.5 * np.cos(q3)
    A = 1.0 + K * np.cos(q2)
    B = 1.0 + 1.5 * np.sin(q3)

    return np.array(
        [
            A * np.cos(q1) - B * np.sin(q1),
            A * np.sin(q1) + B * np.cos(q1),
            -K * np.sin(q2),
        ]
    )


def wrap_angles(q):
    """Map joint angles to [-pi, pi)."""
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


def angular_distance(q1, q2):
    """Euclidean distance between two angle vectors, modulo 2*pi."""
    difference = (np.asarray(q1) - np.asarray(q2) + np.pi) % (2 * np.pi) - np.pi
    return np.linalg.norm(difference)


# ---------------------------------------------------------------------------
# Check the derived position equations against RTB forward kinematics.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(12)
fk_discrepancies = []

for _ in range(100):
    q = rng.uniform(-np.pi, np.pi, size=3)
    fk_discrepancies.append(
        np.max(np.abs(position_formula(q) - robot.fkine(q).t))
    )

print("Maximum formula-vs-RTB discrepancy over 100 random configurations:")
print(f"{max(fk_discrepancies):.3e}")


# ---------------------------------------------------------------------------
# Symbolic resultant using half-angle variables:
#     u = tan(q2/2), v = tan(q3/2)
# ---------------------------------------------------------------------------

u, v = sp.symbols("u v", real=True)

cos_q2 = (1 - u**2) / (1 + u**2)
sin_q2 = 2 * u / (1 + u**2)
cos_q3 = (1 - v**2) / (1 + v**2)
sin_q3 = 2 * v / (1 + v**2)

K_sym = 2 + sp.Rational(3, 2) * cos_q3
A_sym = 1 + K_sym * cos_q2
B_sym = 1 + sp.Rational(3, 2) * sin_q3

rho_squared_sym = X_TARGET**2 + Y_TARGET**2

# Reduced position equations:
#     A^2 + B^2 = x_target^2 + y_target^2
#     -K sin(q2) = z_target
#
# Clear the half-angle denominators. Exact rationals are retained throughout.
F = sp.cancel(
    (A_sym**2 + B_sym**2 - rho_squared_sym)
    * (1 + u**2)**2
    * (1 + v**2)**2
)

G = sp.cancel(
    (-K_sym * sin_q2 - Z_TARGET)
    * (1 + u**2)
    * (1 + v**2)
)

# Explicitly construct polynomials over QQ so SymPy does not use floating
# coefficient arithmetic in the resultant.
F = sp.Poly(F, u, v, domain=sp.QQ).as_expr()
G = sp.Poly(G, u, v, domain=sp.QQ).as_expr()

resultant = sp.factor(sp.resultant(F, G, u))

print("\nResultant in v:")
sp.pprint(resultant)

# For this target, the real roots come from this quartic. The other factors
# in the resultant are (v^2+1)^2 and (v^2+7)^4, which have no real roots.
quartic = (
    289 * v**4
    - 816 * v**3
    - 670 * v**2
    + 1488 * v
    + 193
)

v_roots = sp.nroots(quartic, n=30, maxsteps=200)


# ---------------------------------------------------------------------------
# Back-substitute the real v roots into G to find u.
# ---------------------------------------------------------------------------

analytic_solutions = []

for v_root in v_roots:
    v_complex = complex(v_root)

    # Keep only real roots of the quartic.
    if abs(v_complex.imag) > 1e-10:
        continue

    v_value = float(v_complex.real)

    # For fixed v, G is a polynomial in u.
    polynomial_u = sp.Poly(G.subs(v, sp.Float(v_value)), u)
    coefficients_u = np.array(
        [float(coefficient) for coefficient in polynomial_u.all_coeffs()]
    )

    for u_root in np.roots(coefficients_u):
        if abs(np.imag(u_root)) > 1e-9:
            continue

        u_value = float(np.real(u_root))

        # Filter candidates against both polynomial equations.
        f_residual = abs(float(F.subs({u: u_value, v: v_value})))
        g_residual = abs(float(G.subs({u: u_value, v: v_value})))

        if f_residual > 1e-5 or g_residual > 1e-5:
            continue

        q2 = 2.0 * np.arctan(u_value)
        q3 = 2.0 * np.arctan(v_value)

        # Recover q1 from the rotated horizontal coordinates:
        # [x, y] = rotation(q1) @ [A, B].
        K_value = 2.0 + 1.5 * np.cos(q3)
        A_value = 1.0 + K_value * np.cos(q2)
        B_value = 1.0 + 1.5 * np.sin(q3)

        q1 = (
            np.arctan2(TARGET[1], TARGET[0])
            - np.arctan2(B_value, A_value)
        )

        q = wrap_angles([q1, q2, q3])

        if not any(angular_distance(q, previous) < 1e-7
                   for previous in analytic_solutions):
            analytic_solutions.append(q)


# Check the q3 = pi case separately. It is not represented by finite
# v = tan(q3/2). For this target, it does not produce an additional solution.
q3_special = np.pi
K_special = 2.0 + 1.5 * np.cos(q3_special)
sin_q2_special = -TARGET[2] / K_special

if abs(sin_q2_special) <= 1.0:
    candidate_q2_values = [
        np.arcsin(sin_q2_special),
        np.pi - np.arcsin(sin_q2_special),
    ]

    for q2_special in candidate_q2_values:
        A_special = 1.0 + K_special * np.cos(q2_special)
        B_special = 1.0 + 1.5 * np.sin(q3_special)

        radial_error = abs(
            A_special**2 + B_special**2 - TARGET[0]**2 - TARGET[1]**2
        )

        if radial_error < 1e-8:
            q1_special = (
                np.arctan2(TARGET[1], TARGET[0])
                - np.arctan2(B_special, A_special)
            )
            q_special = wrap_angles(
                [q1_special, q2_special, q3_special]
            )
            if not any(
                angular_distance(q_special, previous) < 1e-7
                for previous in analytic_solutions
            ):
                analytic_solutions.append(q_special)


# ---------------------------------------------------------------------------
# Multi-start numerical IK using the position Jacobian from RTB.
# ---------------------------------------------------------------------------

numeric_solutions = []

for _ in range(1500):
    initial_guess = rng.uniform(-np.pi, np.pi, size=3)

    result = root(
        fun=lambda q: robot.fkine(q).t - TARGET,
        x0=initial_guess,
        jac=lambda q: robot.jacob0(q)[:3, :],
        method="hybr",
    )

    q = wrap_angles(result.x)
    position_error = np.linalg.norm(robot.fkine(q).t - TARGET)

    if position_error < 1e-7:
        if not any(
            angular_distance(q, previous) < 1e-5
            for previous in numeric_solutions
        ):
            numeric_solutions.append(q)


# ---------------------------------------------------------------------------
# Print and compare both solution sets.
# ---------------------------------------------------------------------------

print(f"\nAnalytic resultant solutions: {len(analytic_solutions)}")
for q in sorted(analytic_solutions, key=lambda angles: angles[0]):
    fk_position = robot.fkine(q).t
    error = np.linalg.norm(fk_position - TARGET)
    print(
        f"q = {np.array2string(q, precision=10)}, "
        f"FK = {np.array2string(fk_position, precision=10)}, "
        f"error = {error:.3e}"
    )

print(f"\nMulti-start Jacobian IK solutions: {len(numeric_solutions)}")
for q in sorted(numeric_solutions, key=lambda angles: angles[0]):
    fk_position = robot.fkine(q).t
    error = np.linalg.norm(fk_position - TARGET)
    print(
        f"q = {np.array2string(q, precision=10)}, "
        f"FK = {np.array2string(fk_position, precision=10)}, "
        f"error = {error:.3e}"
    )

match_errors = []
for q_analytic in analytic_solutions:
    distances = [
        angular_distance(q_analytic, q_numeric)
        for q_numeric in numeric_solutions
    ]
    match_errors.append(min(distances) if distances else np.inf)

print("\nNearest numerical-solution errors for analytic roots:")
print([f"{error:.3e}" for error in match_errors])

all_analytic_match = (
    bool(match_errors) and max(match_errors) < 1e-5
)
same_number_of_solutions = (
    len(analytic_solutions) == len(numeric_solutions)
)

print("\nAll analytic roots match numerical roots:", all_analytic_match)
print("Both methods found the same number of roots:", same_number_of_solutions)