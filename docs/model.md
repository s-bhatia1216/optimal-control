# Model definition and FDR parameter decisions

This file records every modeling choice that required judgment, especially
where the report lists more than one value for the same quantity. Nothing
here is silently assumed: each choice names its FDR source, the alternative,
and the reason.

Working units everywhere: hours, watt-hours, megabits. Power in W times time
in h gives Wh directly; rates in Mbit/h.

## 1. Problem statement

One orbit, t in [0, T], periodic. States: battery energy E [Wh], stored data
D [Mbit]. Controls: science level u_s in [0, 1], transmit level
u_c in [0, c(t)], harvest fraction u_h in [0, 1] (shunt regulator, the
"power dump" needed for feasibility; see audit item 1).

    dE/dt = u_h P_in(t) - P_0(t) - P_s u_s - P_c u_c
    dD/dt = r_hk + r_s u_s - r_c u_c

    E_min <= E <= E_max,  0 <= D <= D_max,  E(T) = E(0),  D(T) = D(0)

    maximize  J = integral w(t) r_s u_s dt,  w(t) = ROI indicator (pluggable)

The only deviation from the kickoff dynamics is the constant housekeeping
inflow `r_hk` (next section). It is additive and control-independent, so it
changes no switching structure in the PMP analysis; it only shifts the data
balance. Set `data.include_housekeeping: false` in a config to remove it.

## 2. Housekeeping data inflow (model extension, flagged for sign-off)

FDR Table 27 shows that the spacecraft generates housekeeping data all orbit
(IMU 812 bps, thermal sensors 88 bps, other housekeeping 250 bps) plus the
always-on FGM survey stream (6 bps): 1156 bps total. This data must be
downlinked (it is in the FDR data budget), so leaving it out would understate
the downlink load by ~100 Mbit/orbit (about 11% of the 1 h contact capacity).
We include it as a constant inflow. ASSUMPTION: FGM stays in survey mode
(6 bps); burst mode (3072 bps) appears only as a stress variant.

## 3. Load decomposition (P_0, P_s, P_c)

The FDR power budget (Tables 85/87) reports mode totals: Day 62.2 W, Eclipse
60.8 W. Our model needs the controllable pieces split out. We rebuild P_0
bottom-up from Table 85 components instead of subtracting from the totals,
because the stated subtotals and their components differ slightly (see 3a).

| term | value | composition (FDR Table 85 unless noted) |
|---|---|---|
| P_0 day | 14.2 W | ADCS 3.5 + flight computer 7.0 + FGM 0.8 + thermal sensors 1.0 + radio idle 1.9 (Table 40) |
| P_0 eclipse | 59.0 W | ADCS 3.0 + FC 7.0 + FGM 0.8 + heaters and sensors 46.3 + radio idle 1.9 |
| P_s | 3.7 W | ESA 1.7 + EPD/EIS+ 2.0 (also Table 20) |
| P_c | 68.3 W | power amplifier 53.2 + radio TX minus idle (17.0 - 1.9 = 15.1, Table 40) |

Notes recorded while deriving this:

- (3a) Table 85's Day OPL/CCD subtotal is 57.7 W, but its own components sum
  to 64.7 W (53.2 + 2.0 + 1.7 + 0.8 + 7.0): the stated subtotal appears
  not to include the 7.0 W flight computer, while the Eclipse subtotal
  (11.5 W) does include it. We
  include the flight computer in P_0 at all times (it must run).
- (3b) Table 85's Eclipse column lists ESA (1.7 W) and EPD (2.0 W) ON, while
  Tables 13/14 say Eclipse mode powers them OFF. So the power budget already
  carries the power for eclipse science that the mode tables turn off.
  We treat science in eclipse as a control decision, which is the point of the
  project. The baseline policy follows Tables 13/14 (off).
- (3c) The 1.9 W radio idle draw (Table 40) appears nowhere in the Table 85
  budget. We include it in P_0; with it, our eclipse load with science on is
  62.7 W vs the FDR's 60.8 W, and the difference is exactly this 1.9 W.
- (3d) ASSUMPTION: the power amplifier (53.2 W in Table 85, listed under the
  Day column because downlink happens in sunlight) draws power only while
  transmitting, so it belongs to P_c, not P_0. Physically it is keyed with
  the transmitter.
- (3e) Battery recharge power (29.6 W in Table 87) is not a load: recharge is
  what the E dynamics do. It must not be double counted.

## 4. Battery capacity and bounds

The report gives three different sets of numbers for the battery:

- Section 10.7.4 and Table 90: 8s6p Molicel INR-21700-M50A, 28.8 V bus,
  4.8 Ah cells -> 48 x 3.6 V x 4.8 Ah = 829.44 Wh nominal. Table 90 also
  gives BOL required capacity 28.1 Ah (809 Wh) and EOL 25.5 Ah (734.4 Wh).
- Section 10.7.4.1 text: "72.75 Ah BOL" with a "35% degradation" argument.
  At 28.8 V that is 2095 Wh, which does not follow from the 6-parallel cell
  configuration (28.8 Ah). We use the hardware configuration.

DECISION: capacity = 829.44 Wh (nameplate of the flown configuration),
EOL variant = 734.4 Wh (Table 90 EOL, about 11.5% fade). Both appear in
configs; the audit sweeps them.

DoD floor: the report lists two values: 60% DoD in Table 14 and Section
10.7.4.2 (E_min = 0.4 cap) and 30% in Table 13 (E_min = 0.7 cap).
DECISION: 60% nominal (it is the design basis used to size the battery),
30% kept as a stress variant. Audit observation: with the worst-case
4.356 h eclipse, the 30% floor and the eclipse load cannot both be
satisfied, even with the payload off (257 Wh needed vs 248.8 / 220.3 Wh
usable at BOL / EOL).

E_max = capacity (full charge). ASSUMPTION: charge/discharge efficiency = 1
and no taper near full charge; losses are second order for this study and
identical across policies being compared.

## 5. Orbit, windows, and timeline

- Group 1 orbit (worst case): r_p = 1.1588 R_E, r_a = 12.09 R_E (Table 46).
  Period recomputed from Kepler's third law: 24.009 h (FDR says 24 h; <0.04%
  apart). ROI crossing times computed in closed form from Kepler's equation;
  resulting ROI fraction 79.74% vs FDR's 79.73%.
- t = 0 at perigee. The contact window wraps the period boundary; all window
  logic uses circular time distance. Periodic BCs make the origin a plotting
  convention only.
- Eclipse: duration and center configurable; nominal 4.356 h centered on
  apogee (Table 47 worst case, tail season). P_in = 143.6 W sunlit, 0 in
  eclipse (no penumbra ramp: ASSUMPTION, penumbra is ~15 min total per
  Table 47 and affects both policies equally).
- Contact: 1 h centered on perigee (Sec 5.7.10, Table 13). Note: Table 46
  lists "time below 1.08 R_E" (about 1 h) as the comm window even
  though r_p = 1.1588 R_E > 1.08 R_E, so that criterion is not met anywhere
  on the orbit. We treat contact as exactly 1 h centered on perigee
  and note that Table 42's link budget closes at the 12 R_E slant range, so
  the 1 h window is a scheduling choice, not a geometry limit.

## 6. Data rates and storage

- r_s = ESA 5632 + EPD 670 = 6302 bps = 22.69 Mbit/h at u_s = 1 (Table 27).
- r_hk = 1156 bps = 4.16 Mbit/h (Sec 2 above).
- r_c = 253394.4928 bps = 912.2 Mbit/h (Table 42 downlink rate; this value
  already carries the FDR's 1.5x margin on generated data). The link-budget
  comparison tables (36-38) use 243.4 kbps instead; we use Table 42 because
  it is the rate the final link budget actually closes at. Swapping the
  conservative value into the config shifts the downlink capacity by 4%.
- D_max = 4 GB SSR (Sec 5.7.10) = 32000 Mbit using decimal GB (4e9 bytes x 8).
  Binary GiB would give 34360 Mbit; immaterial since D stays below 2% of
  either at nominal.
- The baseline transmits only while backlog exists. When the buffer empties
  mid-contact the radio sends fill data (FDR Table 34 has a fill-data virtual
  channel); in the model this is u_c dropping to the pass-through level
  (r_hk + r_s u_s)/r_c, which keeps D = 0.

## 7. Objective

J = integral of w(t) r_s u_s dt = science megabits collected inside the ROI,
with w the ROI indicator. All science inside the ROI is weighted equally
(no extra value at apogee). `OrbitTimeline.w()` is the single place to change
if later experiments up-weight apogee. Implemented as minimizing -J.

## 8. Known limitations (accepted for scope)

- Two-body orbit, fixed worst-case eclipse geometry; no seasonal variation
  within a single solve (sensitivity sweeps vary it across solves).
- No battery thermal coupling (heater load is exogenous in P_0).
- Controls modeled as continuous levels; the FDR's instruments are on/off,
  but a fractional u_s is realizable as duty cycling within a time step and
  the PMP analysis predicts where interior values appear (boundary arcs).
