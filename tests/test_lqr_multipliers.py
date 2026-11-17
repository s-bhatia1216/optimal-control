"""Validate costate extraction from IPOPT multipliers on a known problem.

Toy problem: scalar finite-horizon LQR,

    xdot = a x + b u,   J = int_0^T (q/2) x^2 + (r/2) u^2 dt + (qT/2) x(T)^2,

whose costate (classic minimum convention, H_c = L + p f) is p(t) = P(t) x(t)
with P from the Riccati ODE -Pdot = 2 a P + q - b^2 P^2 / r, P(T) = qT.

We transcribe with EXACTLY the machinery used for MATRIX (trapezoidal defects,
piecewise-constant controls, CasADi Opti + IPOPT) and check the documented
convention (docs/direct_method.md):

    defect_k :  x_{k+1} - x_k - (h/2)(f_k + f_{k+1}) = 0
    dual     :  lambda_k = -p_classic(t_{k+1/2})      [NO h scaling]
                         = +p_Liberzon(t_{k+1/2})     [H maximized, p0 = -1]

and that the dual of the initial-condition constraint is -p_classic(0).
"""

import casadi as ca
import numpy as np
import pytest
from scipy.integrate import solve_ivp

# Documented empirical conventions (determined here, consumed by direct.py).
DEFECT_DUAL_IS_MINUS_CLASSIC_COSTATE = True
LIBERZON_COSTATE_SIGN = +1  # p_Liberzon = LIBERZON_COSTATE_SIGN * lambda_ipopt

A, B, Q, R, QT, X0, T = -0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0


def solve_lqr_collocation(N: int):
    """Trapezoidal-collocation LQR solve; returns states, duals, grid."""
    h = T / N
    opti = ca.Opti()
    x = opti.variable(N + 1)
    u = opti.variable(N)

    f = lambda xx, uu: A * xx + B * uu
    c_def = (x[1:] - x[:-1] - h / 2 * (f(x[:-1], u) + f(x[1:], u))) == 0
    opti.subject_to(c_def)
    c_init = x[0] == X0
    opti.subject_to(c_init)

    # Trapezoid in the state cost, exact in the piecewise-constant control cost.
    obj = (h * Q / 4) * (ca.sumsqr(x[:-1]) + ca.sumsqr(x[1:])) \
        + (h * R / 2) * ca.sumsqr(u) + QT / 2 * x[N] ** 2
    opti.minimize(obj)
    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0, "sb": "yes"})
    sol = opti.solve()

    return {
        "x": np.array(sol.value(x)).flatten(),
        "lam_defect": np.array(sol.value(opti.dual(c_def))).flatten(),
        "lam_init": float(sol.value(opti.dual(c_init))),
        "t_mid": (np.arange(N) + 0.5) * h,
    }


def riccati_costate(t, x_of_t):
    """Classic costate p(t) = P(t) x(t) from the backward Riccati ODE."""
    sol_P = solve_ivp(
        lambda tt, P: -(2 * A * P + Q - B**2 * P**2 / R),
        [T, 0.0], [QT], dense_output=True, rtol=1e-10, atol=1e-12,
    )
    return sol_P.sol(t)[0] * x_of_t, sol_P


@pytest.mark.parametrize("N", [200, 400])
def test_defect_duals_match_riccati_costate(N):
    out = solve_lqr_collocation(N)
    x_mid = 0.5 * (out["x"][:-1] + out["x"][1:])
    p_classic, _ = riccati_costate(out["t_mid"], x_mid)

    scale = np.max(np.abs(p_classic))
    err_documented = np.max(np.abs(out["lam_defect"] + p_classic)) / scale
    err_wrong_sign = np.max(np.abs(out["lam_defect"] - p_classic)) / scale

    # The documented convention must hold tightly; the opposite sign must not.
    assert err_documented < 1e-3, "lambda_k should equal -p_classic(t_mid)"
    assert err_wrong_sign > 0.5, "sign convention flipped from documented one"


def test_no_timestep_scaling_needed():
    """Duals are O(1) costates for defects in difference form: halving h must
    leave the dual magnitude unchanged (not halve/double it)."""
    lam_200 = solve_lqr_collocation(200)["lam_defect"]
    lam_400 = solve_lqr_collocation(400)["lam_defect"]
    assert np.abs(lam_200[0] / lam_400[0] - 1.0) < 1e-2


def test_initial_condition_dual_is_minus_p0():
    out = solve_lqr_collocation(400)
    p0_classic, _ = riccati_costate(np.array([0.0]), np.array([X0]))
    assert out["lam_init"] == pytest.approx(-float(p0_classic[0]), rel=1e-3)
