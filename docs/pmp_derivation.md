# PMP analysis of the MATRIX per-orbit problem

Conventions. This document follows the course lecture notes (Sections 16-18,
Pontryagin's Maximum Principle Parts I-III) and Liberzon (2012): the control
Hamiltonian is

    H(x, u, p, p0) = <p, f(x, u, t)> + p0 L(x, u, t),

with abnormal multiplier p0 <= 0 normalized to p0 = -1 in the normal case,
H MAXIMIZED pointwise over the admissible control set, and canonical
equations p_dot = -grad_x H. The problem here is normal: a feasible
trajectory exists with every state constraint strictly inactive (harvest
less, do little science, keep E mid-range and D small), so the constraint
system satisfies a Slater condition and p0 = 0 can be excluded. The lecture
notes do not cover state inequality constraints; for those arcs we use the
direct-adjoining form surveyed in Hartl, Sethi and Vickson (1995) and
Bryson and Ho (1975, ch. 3). All numerical statements below are verified
along the N = 2880 direct solutions by scripts/run_phase3.py, which writes
results/<case>/pmp_checks.{json,md}; the dual-to-costate mapping was
validated separately on an LQR problem (docs/direct_method.md, Section 2).

## 1. Problem

States x = (E, D), controls u = (u_s, u_c, u_h) with
0 <= u_s <= chi_ROI(t), 0 <= u_c <= c(t), 0 <= u_h <= 1. Dynamics

    E_dot = u_h P_in(t) - P_0(t) - P_s u_s - P_c u_c  =: f_E
    D_dot = r_hk + r_s u_s - r_c u_c                  =: f_D

state constraints E_min <= E <= E_max, 0 <= D <= D_max, periodic boundary
conditions x(0) = x(T), and running cost L = -w(t) r_s u_s (we minimize the
negative of the science value). Note f and L do not depend on x, and both
are affine in u. The final time T is fixed and the problem is time varying
through P_in, P_0, c and (in the apogee-weighted cases) w, so H is not
conserved globally; see Section 6.

## 2. Hamiltonian, costates, switching functions

With p = (p_E, p_D) and p0 = -1,

    H = p_E f_E + p_D f_D + w(t) r_s u_s
      = [terms independent of u]
        + theta_s(t) u_s + theta_c(t) u_c + theta_h(t) u_h,

with the switching functions (all in Mbit/h)

    theta_s = r_s (w(t) + p_D) - P_s p_E
    theta_c = -(P_c p_E + r_c p_D)
    theta_h = P_in(t) p_E.

Because f and L are independent of x, the canonical equation on any arc
where no state constraint is active is simply

    p_E_dot = -dH/dE = 0,    p_D_dot = -dH/dD = 0,

so the costates are PIECEWISE CONSTANT, changing only where a state
constraint is active. This is the cleanest structural prediction of the
model and it is exactly what the multiplier-based costates show.

## 3. Hamiltonian maximization: bang-bang law and its verification

H is linear in u on a box, so (lecture notes, Section 18.6)

    u_i = upper bound   if theta_i > 0,
    u_i = 0             if theta_i < 0,
    u_i undetermined    if theta_i = 0 on an interval (singular arc).

Verified per interval along every case (results/<case>/pmp_checks.md):
on all intervals where |theta_i| exceeds a tolerance of 1e-5 times its
dynamic range, the control sits on the predicted bound with zero violation
(agreement 100.00% in every case, worst deviation 0 at solver precision).
Where |theta_i| is below tolerance the law is silent; those singular
intervals are cataloged in Section 5.

Case structure implied by the signs:

- Nominal: all state constraints slack except E_max, so p_E = p_D = 0 a.e.
  Then theta_s = r_s w > 0 on the whole ROI (science always on there),
  theta_c = 0 (downlink slack: transmit timing is free), theta_h = 0
  (Section 4). Verified: 2300 of 2300 ROI intervals decided and correct.
- S1 (tight battery): p_E > 0 across the eclipse (Section 4), rationing
  eclipse science. Uniform w makes the whole eclipse singular; apogee w
  turns it bang-bang with threshold w = P_s p_E / r_s (Section 5).
- S2 (short contact): downlink binds, p_D < 0, theta_c = -r_c p_D > 0, so
  u_c saturates the entire contact window. Uniform w: theta_s = r_s (1 +
  p_D) = 0 with p_D = -1, the whole ROI is singular. Apogee w: science on
  exactly where w(t) > -p_D (Section 5).

## 4. State-constrained arcs

### 4.1 The E_max boundary arc carries an interior control

On a sunlit arc with E = E_max, staying on the boundary forces E_dot = 0:

    u_h = (P_0 + P_s u_s + P_c u_c) / P_in   (strictly interior: ~0.1-0.5).

This is the guaranteed non-bang-bang arc of the problem. It is consistent
with the maximum principle because theta_h vanishes there: on the boundary
arc the constraint multiplier absorbs dH/dE, and maximization of H in u_h
must not push off the boundary, which requires p_E = 0 on the arc, hence
theta_h = P_in p_E = 0. Numerically theta_h is identically zero in every
case (max |theta_h| about 1e-7): energy in sunlight is free because excess
must be dumped anyway. The interior-control law is verified on the polished
trajectory (which rides E_max; the stage-1 analytic center drifts below it):
e.g. s1_uniform, 2095 sunlit intervals (17.46 h) on the bound with max
residual about 1e-3 in u_h, 100% strictly interior.

### 4.2 The E_min constraint can only touch, not hold

In eclipse P_in = 0, so E_dot = -P_0 - P_s u_s - P_c u_c <= -P0_eclipse
= -59.0 W < 0 for every admissible control. A sustained E = E_min arc would
need E_dot = 0, so none can exist inside the eclipse; the constraint is met
at an isolated TOUCH POINT, which the solutions place exactly at eclipse
exit (the last moment of discharge).

### 4.3 Costate jumps at the two junctions

With state constraints, p may jump at junctions; the jump is proportional
to the constraint gradient with a nonnegative multiplier (Hartl, Sethi and
Vickson 1995; Bryson and Ho 1975, ch. 3). The transcription makes this
exact and sign-unambiguous: KKT stationarity of the LP with respect to an
interior state node k gives the discrete identity

    p(t_k+) - p(t_k-) = mu_upper(k) - mu_lower(k),

where mu_upper, mu_lower are the (nonnegative) duals of the upper and lower
state bounds at that node. Fitted over all nodes and all cases the identity
holds with max residual about 1e-13 (pmp_checks.md), i.e. the costate is
constant wherever both bounds are slack and jumps only through the bound
multipliers.

S1 has both junction types in one orbit, with E usable exactly equal to the
eclipse draw at the rationed science level:

- Eclipse entry, t = 9.826 h: the trajectory LEAVES the E_max arc with the
  battery exactly full (the constraint is pointwise active at this node).
  p_E jumps 0 -> r_s/P_s = +6.1317 Mbit/Wh, and mu_Emax at that node is
  +6.1317: the jump equals the E_max multiplier point mass.
- Eclipse exit, t = 14.182 h: the E_min touch. p_E jumps r_s/P_s -> 0, and
  mu_Emin at the touch node is +6.1317: the jump equals the E_min
  multiplier point mass.

Between the junctions p_E = r_s/P_s = 6.1317 Mbit/Wh exactly: one stored
watt-hour at eclipse entry buys 1/P_s hours of science worth r_s/P_s
megabits. Outside the arc p_E = 0 (energy is free in sunlight).

## 5. Singular arcs

A singular arc is an interval where a switching function vanishes
identically, so the maximum condition does not determine the control
(lecture notes, Section 18.6.2; Liberzon 2012). Three appear here, all
verified with |theta| below 1e-8 on the arc:

- S1, uniform w: on the eclipse arc theta_s = r_s (1 + p_D) - P_s p_E with
  p_D = 0 and p_E = r_s/P_s, so theta_s = 0 identically: the TOTAL eclipse
  science is pinned by the energy budget (1.995 h) but its placement is
  free. The transcribed problem has a flat optimal face; IPOPT returns its
  analytic center (u_s about 0.46 smeared across the eclipse,
  fig3_stage1_solution), and the polish stage picks the latest-possible
  block, another member of the same family. 524 singular intervals, max
  |theta_s| = 4e-9.
- S2, uniform w: p_D = -1 makes theta_s = 0 on the WHOLE ROI: the downlink
  cap fixes total science (15.700 h) but not which hours are dropped. All
  2300 ROI intervals singular.
- theta_c in nominal and S1: downlink is slack and p_E = p_D = 0, so
  transmit timing inside the contact window is free.

The apogee weight w(t) = chi_ROI r(t)/r_a is the designed tie-breaker: with
w time varying, theta_s = 0 can hold only where w(t) equals a constant
threshold, i.e. at isolated times, so the singular arc collapses to a
bang-bang block with a marginal-value switching condition:

- S1 apogee: science on where w(t) > P_s p_E / r_s = 0.99538; the solved
  switches at 11.006 and 13.003 h have w = 0.99536 (match to 2e-5). The
  block is centered on apogee.
- S2 apogee: science on where w(t) > -p_D = 0.68325; the solved switches at
  4.154 and 19.855 h have w = 0.6831 (match to 2e-4), radius 8.26 R_E. The
  stored-data shadow price equals the science value at the marginal dropped
  time.

In both cases the remaining "singular" count is 2 intervals, the mesh cells
that straddle the switch times.

## 6. Transversality and the Hamiltonian

Periodic boundary conditions x(0) = x(T) with both endpoints free on the
constraint x(0) - x(T) = 0 give the periodic transversality condition

    p(0) = p(T)

(the terminal-manifold condition of the lecture notes, Section 18.3,
applied to the diagonal manifold). Verified: the endpoint costate gaps are
below 1e-15 in every case, and the dual of the discrete periodicity
constraint equals the endpoint costate to the same precision.

The problem is time varying (P_in, P_0, c, and w in the apogee cases), so
H is only piecewise conserved: d/dt H|_* equals the explicit partial
derivative of H with respect to t (lecture notes, Section 18.2 for the
autonomous case). Verified: across interior nodes that are neither window
edges nor costate junctions, the jump of H minus the explicit-time
contribution is at most 6e-8 Mbit/h for the uniform-w cases and at most
6e-3 Mbit/h (an O(h) effect at the interior switches, about 3e-4 of the H
scale) for the apogee cases. H is continuous across control switches, as
required (lecture notes, Section 18.1).

## 7. What this buys the paper

The direct solution and the maximum principle validate each other in both
directions. PMP explains every feature the LP produces (piecewise-constant
costates with jumps equal to bound multipliers, the interior-control E_max
arc, the E_min touch point, the singular smears and their apogee-weighted
resolution, the marginal-value switch thresholds), and the direct solution
supplies the multipliers that make every PMP quantity computable without
solving a boundary-value problem. The quantitative checks are reproducible
via scripts/run_phase3.py and summarized in results/checkpoint3.md.

References: course lecture notes Sections 16-18 (Pontryagin's Maximum
Principle, Parts I-III); D. Liberzon, Calculus of Variations and Optimal
Control Theory, Princeton University Press, 2012; R. F. Hartl, S. P. Sethi
and R. G. Vickson, "A survey of the maximum principles for optimal control
problems with state constraints", SIAM Review 37(2), 1995; A. E. Bryson and
Y.-C. Ho, Applied Optimal Control, Hemisphere, 1975, ch. 3.
