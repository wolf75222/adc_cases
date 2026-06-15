# Scoping note: the MATLAB BGK collision (collision15.m)

ADC-277, step 1 (scoping only). This note records the evidence behind the decision in the
hyqmom15 README ("MATLAB fidelity: the BGK collision"). It ports no code. The architecture
choice is gated on the ADC-273 multi-agent vote.

## What collision15.m does

File `collision15.m` in RIEMOM2D, 43 lines, signature `Mout = collision15(M, dt, Kn)`. It is
the elastic BGK collision applied per cell as a fractional step after the transport update:

```matlab
Theta  = (C20 + C02)/2;           % 2-D granular temperature from the central moments
sTheta = sqrt(Theta);
CG20 = Theta; CG11 = 0; CG02 = Theta;   % Maxwellian (isotropic) covariance
tc  = Kn/(rho*sTheta*2);          % collision time scale
MG  = InitializeM4_15(rho, umean, vmean, CG20, CG11, CG02);  % local Maxwellian moments
Mout = MG - exp(-dt/tc)*(MG - M); % semi-analytical BGK update
```

Properties:

- target: the local Maxwellian `MG` of the same density `rho`, mean velocity `(umean, vmean)`
  and temperature `Theta`. The covariance is forced isotropic (`CG20 = CG02 = Theta`,
  `CG11 = 0`), so this is the Maxwellian, not the more general BGK covariance (the latter is
  written in the source as commented-out lines).
- relaxation: `Mout = MG - exp(-dt/tc)(MG - M)`, exact in `dt` for a frozen `tc`. The retention
  factor `exp(-dt/tc)` is in `(0, 1)`: the state moves from `M` toward `MG` and never overshoots.
- conservation: `rho`, the mean velocity and the temperature `Theta` define `MG`, and the update
  is an affine blend of `M` and `MG`, so the low-order moments are preserved to the precision of
  `InitializeM4_15`. The higher orders relax toward equilibrium.
- time scale: `tc = Kn / (2 rho sqrt(Theta))`. Large `Kn` gives a long `tc` (nearly
  collisionless, `exp(-dt/tc) -> 1`, `Mout -> M`); small `Kn` gives a short `tc` (strong
  relaxation, `Mout -> MG`).

`collision15.m` is distinct from `collision15_anisotropic.m`, which is the realizability
projector (signature `(s03..s40, lamin)`, no time scale, called by `relaxation15.m`). The
projector is already ported in `relaxation.py`; the BGK collision is not.

## Where MATLAB activates it, and at which Kn

The collision is gated by `if Kn <= 10` in every main script. In all three shipped scripts
`Kn` is hard-coded to `Kn = 1000/1` (so `Kn = 1000`), which never satisfies the gate. As
shipped, RIEMOM2D runs collisionless.

| MATLAB main script | `Kn` definition | collision call site | gate result |
|---|---|---|---|
| `main_electrostatic_wave.m` | line 49: `Kn = 1000/1;` | line 438: `MMC = collision15(MM,dt,Kn);` | line 431 `if Kn <= 10`: never taken |
| `main_pb_2Dcrossing_2DHyQMOM15.m` | line 47: `Kn = 1000/1;` | line 301: `MMC = collision15(MM,dt,Kn);` | line 294 `if Kn <= 10`: never taken |
| `main_x_shock_tube.m` | line 58: `Kn = 1000/1;` | line 331: `MMC = collision15(MM,dt,Kn);` | line 324 `if Kn <= 10`: never taken |

A repository-wide grep finds no script that sets `Kn` to any value other than `1000/1`
(the only other occurrences of a small number near `Kn` are in output-filename templates and
in `compute_dt.m` style comments, not assignments).

## Are the targeted campaigns collisional?

No. The two campaigns this case reproduces are exactly the gated-off branches:

- diocotron: `main_electrostatic_wave.m`, which calls `initialize_dicotron` at line 195. The
  Python port is `run_diocotron.py` (its header cites `main_electrostatic_wave.m, section
  dicotron`). `Kn = 1000` there, collision off.
- crossing jets: `main_pb_2Dcrossing_2DHyQMOM15.m`. The Python port is `run_crossing.py`
  (header cites `main_pb_2Dcrossing_2DHyQMOM15.m`). `Kn = 1000` there, collision off.

`initialize_dicotron.m` contains no `Kn` and no collision call: the diocotron initial condition
is purely the perturbed ring plus the ExB drift. So the targeted physics is quasi-collisionless,
and the current validations match the reference at `Kn = 1000`.

## Decision

Out of short-term scope, deferred. Justification:

1. it blocks no current validation: the targeted regimes never enter the `Kn <= 10` branch;
2. the operator form is structural (it touches the post-step ordering relative to
   `relaxation15`, the time-step policy, and possibly a future compiled path), so per ADC-273 it
   needs the multi-agent vote before any implementation PR. ADC-273 already lists "Port eventuel
   de collision15.m / BGK" among its gated chantiers.

Candidate operator forms for that vote (not decided here):

- explicit local source: per-cell post-step overwrite `Mout = MG - exp(-dt/tc)(MG - M)`, the
  literal MATLAB transcription. Simplest, exact in `dt` per cell, but it inherits the MATLAB
  fractional-step ordering (after transport, after `relaxation15`).
- implicit / IMEX: only worth it if a stiff small-`Kn` regime enters scope, where an explicit
  step would constrain `dt`. The MATLAB form is already stable in `dt` for a frozen `tc`, so the
  gain is marginal unless `tc` varies fast within a step.
- projector-like local overwrite: misfits the semantics. BGK is a smooth relaxation with
  retention `exp(-dt/tc)`, not a hard projection onto a set, so reusing the realizability
  projector pattern would change behavior.
- compiled native brick (adc_cpp): the only form fit for GPU campaigns, the heaviest, and it
  would have to fix its ordering against the future compiled `relaxation15`.

If the BGK port is later retained, step 2 of ADC-277 applies: port to a Python oracle or C++
native, validate cell by cell against Octave at a `Kn <= 10` value, add a short run with
collision on, and document the ordering, conservation, realizability and stability relative to
`relaxation15`.

## Provenance

RIEMOM2D inspected at `/Users/romaindespoulain/Documents/RIEMOM2D` (git checkout dated
2026-06-11). Files cited: `collision15.m`, `collision15_anisotropic.m`,
`main_electrostatic_wave.m`, `main_pb_2Dcrossing_2DHyQMOM15.m`, `main_x_shock_tube.m`,
`initialize_dicotron.m`, `relaxation15.m`, `InitializeM4_15.m`. Knudsen value found: `Kn = 1000`
in all three main scripts; collision gate `if Kn <= 10` never taken.
