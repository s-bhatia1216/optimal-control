# Direct method: transcription, dual conventions, and what to expect

This documents the Phase-2 direct solution (src/matrix_ocp/direct.py): how the
continuous problem becomes a finite NLP, how IPOPT multipliers map to PMP
costates, and which features of the solutions are real structure versus
solver selection on degenerate directions.

## 1. Transcription

- Mesh: N uniform intervals over [0, T], then the six window edge times (ROI
  entry/exit, eclipse start/end, contact edges) are inserted as extra nodes,
  so every interval lies purely inside or outside each window. The reported
  "N" is the uniform base count; the actual interval count is N + about 6.
- Variables: states E, D at nodes; controls u_s, u_c, u_h piecewise constant
  on intervals. Exogenous inputs (P_in, P_0, contact bound c, weight w) are
  averaged per interval, which on the snapped mesh makes them exact 0/1
  (or exact w averages) per interval.
- Defects: x_{k+1} - x_k - h_k f_k = 0. Because the dynamics are affine in
  the controls and do not depend on the states, this one-step rule equals the
  exact integral of the continuous dynamics over the interval. There is no
  integration error at all; the ONLY restriction the mesh imposes is that
  controls cannot switch inside an interval.
- Consequence seen in the refinement study (results/*/refinement.md): the
  objective is constant in N to seven digits for every case. Cases whose
  optimal switches sit at window edges (all uniform-w cases) are exact at any
  N; the apogee-weighted cases have interior switch times (the eclipse
  science window in S1, the ROI-edge drop in S2) that move by O(h) across
  meshes while the objective moves only in its fifth significant digit,
  consistent with an O(h^2) value penalty for an O(h) switch-time error.
- Scaling: hours / Wh / Mbit keep all quantities within about three orders
  of magnitude of one; IPOPT converges in 20-60 iterations on every case.

The transcribed problem is a LINEAR program (linear dynamics, bounds, and
objective). We still solve it with IPOPT via CasADi Opti because we want the
NLP machinery (and its duals) that generalizes if the model stops being
linear, but the LP structure explains the solution behavior in Section 3.

## 2. From IPOPT multipliers to PMP costates

Validated in tests/test_lqr_multipliers.py on a scalar finite-horizon LQR
against the Riccati costate p_classic(t) = P(t) x(t); max relative error
about 1e-5 at N = 400.

With defects written as above and the cost MINIMIZED, the CasADi/IPOPT dual
of defect k satisfies

    lambda_k = - p_classic(t_{k+1/2})          (classic convention,
                                                H_c = L + <p, f> minimized)

with NO timestep scaling: the h sits inside the defect, so the dual is an
O(1) costate sample at the interval midpoint (test_no_timestep_scaling
confirms the magnitude is mesh-independent). The dual of the initial-state
constraint is -p_classic(0).

The PMP write-up (Phase 3) uses the Liberzon (2012) convention: for a
minimum-cost problem, H = <p, f> + p0 L with p0 = -1 in the normal case, and
H is maximized; then p_Liberzon = -p_classic. Chaining the two signs:

    p_Liberzon(t_{k+1/2}) = + lambda_k^{IPOPT}.

direct.py therefore stores p_E, p_D as the raw defect duals
(LIBERZON_COSTATE_SIGN = +1), defined at interval midpoints.

State-bound duals (mu_Emax, mu_Emin, mu_Dmax, mu_Dmin) are kept per node.
They are the discrete counterpart of the measure d(mu) in the state-
constrained maximum principle: nonzero only where the bound is active, and
the jump of the costate across a junction equals the accumulated bound dual
there (checked quantitatively in Phase 3).

Spot checks already wired into tests/test_direct.py:

- S1 (uniform): on the rationed eclipse arc p_E = r_s / P_s = 6.1317 Mbit/Wh
  to 0.1%, and p_E = 0 outside the [eclipse entry, E_min touch] arc.
- S2 (uniform): p_D = -1 (a stored Mbit costs exactly one science Mbit at
  the margin when the rationed science has unit weight).
- S2 (apogee): p_D = -0.683 = -w at the marginal dropped time, i.e. the
  radius at the science switch over r_a.

## 3. Degeneracy: real structure, not solver noise

With a linear objective in u_s only, several directions do not affect the
objective at all. An interior-point method converges to the analytic center
of the optimal face, so these show up as SMOOTH INTERIOR control values, not
bang-bang chatter. Per the checkpoint-1 decisions we report them rather than
regularize them away:

- Singular science arcs. S1 with uniform w: the total eclipse science is
  pinned by the energy budget at 1.995 h, but its placement is free; IPOPT
  returns u_s at about 0.46 across the whole eclipse (the centered spread of
  the ration). The switching function is identically zero there (Phase 3).
  The apogee weight breaks the tie: a clean ON block centered on apogee.
- Worthless off-ROI science (nominal, S1): w = 0 and no binding power or
  data cost makes u_s outside the ROI completely free; IPOPT parks it near
  0.5. In S2 the data cost makes it strictly suboptimal and it vanishes.
  Metrics therefore always report ROI science, and switch-time extraction
  masks u_s by the ROI indicator.
- Harvest timing (u_h) and the D offset. Only u_h's integral between
  E-constraint activations matters, and D is defined up to an additive
  constant when neither data bound binds; IPOPT centers both (E drifts
  between bounds instead of riding E_max; D sits near D_max/2). The baseline
  overlay in the figures shows a representative member of the same optimal
  family that the flight rule would pick.

## 4. Known limitations

- Uniform-mesh switch resolution is O(h) for interior (non-window) switch
  times; the refinement tables quantify it at a few milli-hours by N = 2880.
- IPOPT tolerance 1e-9 leaves duals with about 1e-9 noise, visible in the
  p_D panel of cases where the true value is 0.
- Dual non-uniqueness: where the primal is degenerate the duals of an LP can
  be non-unique as well; the values reported are IPOPT's central ones, and
  the Phase-3 consistency checks are stated against those.
