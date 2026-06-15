# Diocotron sweep: order x resolution x mode (measured, O1/O2 + high-order O5)

This sweep quantifies the diocotron growth-rate gap (numerical `adc` vs Petri's analytical
target) as a function of **resolution** and **reconstruction order**, to decide PR-A
"transport-wall". No physics is changed: the sweep reuses the [`run.py`](run.py) pipeline as-is
(shared ring IC, azimuthal FFT of mode `l` of `phi`, fit of the linear `exp(gamma t)` phase,
normalization by `omega_D`, Petri analytical target in numpy). Script: [`sweep.py`](sweep.py).
Raw data: `out/diocotron/sweep_results.csv`.

This version extends PR-0 (which swept {O1, O2-minmod, O2-vanleer}) with the **high-order axis O5 =
WENO5-Z + SSPRK3**, now reachable from Python (adc_cpp #88, master `ca803dc`). The O5 axis exists to
shed light on the question PR-0 left open: is the l-dependent residual at O2 diffusion (closable by
order) or a structural floor of the cartesian ring boundary?

## Protocol

- **Case**: `adc.System` combining `models.diocotron` + Poisson with a circular conducting wall
  (`wall="circle"`, `wall_radius=0.40`), identical to `run.py`. Ring IC `r0:r1:Rwall =
  0.15:0.20:0.40`, azimuthal perturbation `delta=0.01`, CFL=0.4.
- **Target**: Petri eigenvalues (numpy, `run.py`), invariant in `n`:
  `gamma_3 = 0.772`, `gamma_4 = 0.912`, `gamma_5 = 0.687`.
- **gamma_num**: slope of `log|c_l(phi)|` over the linear phase (existing diagnostic
  `fit_linear_phase`), normalized by `2 pi / rhobar`. `%err = 100 (gamma_num - gamma_ana) / gamma_ana`.
- **Common physical horizon `t_end = 48`**: at fixed `nsteps` the step `dt ~ CFL dx` shrinks with
  `n`, so the final physical time shrinks with `n` (n=256, 900 steps -> t ~ 35, phase not yet
  saturated, gamma over-read). You therefore advance to `t_end = 48` (the horizon of the validated
  `run.py` tuning: n=192, 900 steps -> t ~ 48). At this horizon the sweep **reproduces the README at
  n=192** (l=3 -22 %, l=4 -27 %, l=5 -5 %), which anchors the measurement. This is a loop setting,
  not a new observable.

## Order axis: what is reachable from Python

The order axis required by the spec was {O2, **O5 WENO5-Z**}. At the time of PR-0 (master
`adc_cpp` 30f6dfd), WENO5-Z and SSPRK3 were not reachable from the diocotron case path.
**Since adc_cpp #88 (master `ca803dc`), they are**, via the native `add_block` path:

- `adc.Spatial(limiter="weno5", flux="rusanov")`: WENO5-Z reconstruction (order 5 in smooth zones,
  5-point stencil, 3 ghosts); `make_block` now instantiates the `Weno5` policy. Only the native
  `add_block` path exposes it (the AOT/JIT .so paths allocate 2 ghosts and reject it).
- `adc.Explicit(method="ssprk3")`: SSPRK3 integrator (Shu-Osher, 3 stages, order 3). You pair it
  with WENO5-Z because pairing a high spatial order with SSPRK2 (order 2 in time) would cap the
  effective order. The sweep's order key `weno5` therefore steers both bricks (see `sweep.py`
  `make_system`: `limiter == "weno5"` selects `Explicit(method="ssprk3")`).

Empirical check (master `ca803dc`): `add_block(spatial=adc.Spatial(limiter="weno5"),
time=adc.Explicit(method="ssprk3"))` composes and advances without NaN (the `"weno5"` string, which
raised `System : limiter inconnu 'weno5'` on 30f6dfd, is now accepted). Orders O1/O2 keep the legacy
SSPRK2 integrator (`adc.Explicit()` by default): their rows below reproduce PR-0 exactly (same
observable, same `t_end=48` tuning).

**Order axis actually swept**: `{O1 none, O2 minmod, O2 vanleer, O5 weno5}`. Numerical diffusion
decreases (a) as you raise resolution and (b) as you raise order / lower dissipation
(none -> minmod -> vanleer, then WENO5-Z + SSPRK3 at order 5). The O5 axis is what lets you address
the diffusion-vs-structural question: at order 5 the residual diffusion is strongly bounded, so what
remains at O5 is the most credible candidate for a structural floor.

## Results: gamma_num (%err vs analytical), `t_end = 48`

| n | order | l=3 (target 0.772) | l=4 (target 0.912) | l=5 (target 0.687) |
|---|---|---|---|---|
| 128 | O1 none      | 0.263 (-66.0 %) | 0.005 (-99.5 %) | 0.111 (-83.9 %) |
| 192 | O1 none      | 0.325 (-57.9 %) | 0.371 (-59.3 %) | 0.343 (-50.0 %) |
| 256 | O1 none      | 0.463 (-40.0 %) | 0.452 (-50.5 %) | 0.418 (-39.2 %) |
| 128 | O2 minmod    | 0.506 (-34.4 %) | 0.613 (-32.8 %) | 0.519 (-24.4 %) |
| 192 | O2 minmod    | 0.599 (-22.4 %) | 0.662 (-27.4 %) | 0.652 (-5.1 %) |
| 256 | O2 minmod    | 0.639 (-17.3 %) | 0.801 (-12.1 %) | 0.703 (+2.3 %) |
| 384 | O2 minmod    | 0.679 (-12.1 %) | 0.798 (-12.5 %) | 0.705 (+2.6 %) |
| 512 | O2 minmod    | 0.684 (-11.4 %) | 0.807 (-11.4 %) | 0.727 (+5.9 %) |
| 128 | O2 vanleer   | 0.606 (-21.5 %) | 0.781 (-14.4 %) | 0.685 (-0.4 %) |
| 192 | O2 vanleer   | 0.658 (-14.8 %) | 0.752 (-17.5 %) | 0.714 (+3.9 %) |
| 256 | O2 vanleer   | 0.684 (-11.4 %) | 0.862 (-5.5 %) | 0.744 (+8.3 %) |
| 384 | O2 vanleer   | 0.702 (-9.1 %) | 0.825 (-9.5 %) | 0.710 (+3.3 %) |
| 512 | O2 vanleer   | 0.700 (-9.4 %) | 0.821 (-10.0 %) | 0.721 (+4.9 %) |
| 128 | O5 weno5     | 0.659 (-14.6 %) | 0.874 (-4.2 %)  | 0.735 (+6.9 %) |
| 192 | O5 weno5     | 0.677 (-12.4 %) | 0.768 (-15.8 %) | 0.700 (+1.8 %) |
| 256 | O5 weno5     | 0.692 (-10.3 %) | 0.875 (-4.1 %)  | 0.719 (+4.7 %) |
| 384 | O5 weno5     | 0.706 (-8.6 %)  | 0.828 (-9.2 %)  | 0.702 (+2.2 %) |
| 512 | O5 weno5     | 0.704 (-8.8 %)  | 0.823 (-9.7 %)  | 0.715 (+4.0 %) |

(n=192 O2 minmod = anchor row, reproduces the README. The O1/O2 rows are replayed here and reproduce
PR-0 exactly: in particular n=384 and n=512 O2 on ROMEO return the n=384 O2 values already tabulated
to the hundredth, which anchors the ROMEO build chain on PR-0. O5 = WENO5-Z + SSPRK3, run locally at
n=128/192/256; **rows n=384 and n=512 (all orders) are measured on ROMEO** (x64cpu amd EPYC, SLURM
job 639912, see "run vs skipped"). n=384 O2 = a probe beyond the main grid inherited from PR-0.)

> **Quick read of the high-resolution rows (n=384/512).** The point that changes the reading is
> O5 l=4: at low resolution it was ~ -4 % on both eigen-points (n=128, n=256), which PR-0 read as
> "diffusion almost exhausted". At high resolution O5 l=4 does not fall toward 0 % in these
> measurements:
> it is -9.2 % (n=384) then -9.7 % (n=512). Caveat: both points have a fit window that opens early
> (t0 = 6.3 and 5.4, see traceability table), the exact same flaw as the n=192 point already
> flagged; they are therefore not directly comparable to the two low-resolution eigen-points. The
> cleanest signal is l=3 O5, whose window opens consistently (t0 ~ 6.5) at every n: it goes from
> -10.3 % (n=256) to -8.6 % (n=384) then -8.8 % (n=512), a clear flattening around -9 % at high
> resolution. See the cautious verdict below.

### Traceability: fit window of the O5 l=4 points (low and high resolution)

For each row, the CSV (`out/diocotron/sweep_results.csv`) writes the bounds of the `fit_linear_phase`
window: indices `fit_i0..fit_i1` and times `fit_t0..fit_t1`. Here are those bounds for the O5 l=4
points. The first three (n=128/192/256) are the local PR-0 values; the last two (n=384/512) come from
ROMEO job 639912 (window columns taken directly from the merged CSV `sweep_hires_merged.csv`):

| n | gamma_num | %err | window i0..i1 | window t0..t1 | clean window ? |
|---|---|---|---|---|---|
| 128 | 0.8736 | -4.2 % | 269..528 | 20.8..41.1 | yes (late t0) |
| 192 | 0.7675 | -15.8 % | 108..703 | **5.4**..35.1 | no (early t0) |
| 256 | 0.8745 | -4.1 % | 357..1095 | 13.3..40.8 | yes (late t0) |
| 384 | 0.8282 | -9.2 % | 257..1594 | **6.3**..38.7 | no (early t0) |
| 512 | 0.8229 | -9.7 % | 295..2126 | **5.4**..38.6 | no (early t0) |

Read straight from the window column: the two high-resolution points (n=384, n=512) have a window
that opens early (t0 = 6.3 and 5.4), exactly like the n=192 point already flagged as an artifact
(t0 = 5.4). In other words, the intermediate n=192 point was not an isolated accident: as soon as you
raise n, the `fit_linear_phase` linear window tends to open early for O5 l=4. The only two O5 l=4
points with a late window (n=128 t0=20.8 and n=256 t0=13.3) thus remain the only "clean" ones in the
l=4 set; the new n=384/512 points can neither confirm nor cleanly refute the ~ -4 % value of those
two.

**Why l=3 O5 is the most reliable signal at high resolution.** Unlike l=4, mode l=3 has a fit window
that opens consistently at every n (t0 ~ 6.5: n=384 [t6.5..38.1], n=512 [t6.5..39.9]). Its curve in n
is therefore comparable point by point, without the window bias that affects l=4. It is on l=3 that
the diffusion-vs-structural reading is least ambiguous (see verdict).

## Diffusion-vs-structural reading, per mode

Method: for a given order, if `|%err|` **decreases clearly** with `n` (and with order), the
diffusive part dominates; if it **plateaus** in resolution, the residual is structural (cartesian
ring boundary advected on a full grid, see `docs/PAPER_ROADMAP.md`).

- **l = 3: the cleanest signal. Order reduces the gap, but at high resolution O5 flattens around
  -9 %.** `|%err|` closes with resolution and with order. minmod: -34 % -> -22 % -> -17 % -> -12 %
  -> -11 % (128->512); vanleer (less dissipative): -21 % -> -15 % -> -11 % -> -9 % -> -9 %;
  **O5: -14.6 % -> -12.4 % -> -10.3 % -> -8.6 % -> -8.8 %** (n=128->512). At every n, O5 strictly
  improves on vanleer (the best O2). The new point in this measurement: between n=384 and n=512,
  **the O5 l=3 residual no longer shows a clear closure** (-8.6 % then -8.8 %, a gap within measurement
  noise); and this is the mode whose fit window is consistent at every n (t0 ~ 6.5), so this
  flattening is not a window artifact. **Verdict: on the best-measured mode, order first reduces the
  gap strongly, then O5 `|%err|` appears to plateau around -9 % at high resolution. This suggests a
  residual floor on the order of ~9 % that does not close with resolution (a structural candidate),
  rather than a diffusion still in the process of exhausting itself; remains to be confirmed (a single
  flat step n=384 -> n=512 is not enough to rule out a very slow convergence).**
- **l = 4: the key mode. The -4 % low-resolution O5 does not reproduce at high resolution; but the
  high-resolution points are biased by their window.** At O2, l=4 hits ~10-12 % that does not close
  with resolution (minmod -12.1 % at n=256, -12.5 % at n=384, -11.4 % at n=512; vanleer -5.5 % ->
  -9.5 % -> -10.0 %), which PR-0 read as a structural candidate. PR-0 hoped for the other reading via
  O5 (-4.1 % at n=256, -4.2 % at n=128, "diffusion almost exhausted"). **The ROMEO high-resolution
  measurement does not reproduce this -4 %: O5 l=4 is -9.2 % at n=384 and -9.7 % at n=512**, that
  is, it climbs back into the O2 band (~-10 to -11 %) and the l=3 O5 band (~-9 %) instead of tending
  toward 0. A caveat of first importance: both high-resolution points have a fit window that opens
  early (t0 = 6.3 and 5.4, see traceability table), exactly the flaw that made the n=192 point
  unusable; they therefore probably under-read the slope a bit, like n=192. You cannot, then, assert
  that -9.5 % is the "true" asymptotic value of l=4 O5. What you can say honestly: (a) the -4 %
  observed on the only two clean points (n=128, n=256) does not reproduce at either higher resolution;
  (b) as soon as you raise n, the l=4 window opens early and the measured %err lands in the same
  ~ -9 to -10 % band as l=3 and the O2 orders. **Verdict: PR-0's optimistic reading (l=4 O5 -> ~-4 %,
  hence diffusion) is weakened: it holds only on two clean low-resolution points and does not survive
  the rise in resolution. The n=384/512 points (~-9.5 %) are consistent with a floor on the order of
  ~9-10 % of the same order as l=3, but their early window forbids making a reliable floor measurement
  out of them. l=4 conclusion: no closure toward 0 % at high resolution -> the l=4 residual does not
  behave like a diffusion that exhausts itself; a (structural) floor of ~9-10 % is the most consistent
  candidate, to be confirmed with a robust window diagnostic on l=4 (opening the fit later).**
- **l = 5: weak diffusive part, already resolved; high resolution confirms it.** Already at the O2
  target at n=192 (minmod -5 %, vanleer +4 %), the error crosses zero and stays small and of varying
  sign. **O5: +6.9 % -> +1.8 % -> +4.7 % -> +2.2 % -> +4.0 %** (n=128->512), of the same order of
  magnitude (a few %, varying sign, no clear trend) as the O2 orders (n=512: minmod +5.9 %, vanleer
  +4.9 %). The residual is dominated by measurement noise / slight overshoot, not by a structural
  floor nor a residual diffusion. **Verdict: no notable gap to close; neither order nor high
  resolution reveals a floor on l=5.**

**Overall conclusion (with the O5 axis and the ROMEO high-resolution confirmation).** The n=384/512
measurement (ROMEO job 639912, x64cpu) reverses the optimistic reading PR-0 drew from the two clean
low-resolution points. PR-0 read: "l=4 O5 drops to ~-4 % -> the residual is diffusion, not a floor".
**High resolution does not reproduce this -4 %.** On the two modes where the measurement is usable at
high resolution:

- **l = 3 (best measured, window consistent at every n)**: O5 `|%err|` plateaus around -9 %
  (-10.3 % at n=256, then -8.6 % at n=384 and -8.8 % at n=512, flat between the last two steps).
  This is not the behavior of a diffusion that exhausts itself (which would keep closing), but rather
  that of a candidate residual floor (not yet definitive proof).
- **l = 4**: the -4 % low-resolution O5 does not reproduce at either higher resolution (O5 l=4 =
  -9.2 % at n=384, -9.7 % at n=512). It climbs back into the same ~ -9 to -10 % band as l=3 and the
  O2 orders. The major reservation: both l=4 points have an early fit window (t0 = 6.3 and 5.4, like
  the n=192 point already discarded), so you cannot make a reliable floor value out of them; you can
  only say that l=4 does not tend toward 0 % in these high-resolution measurements.
- **l = 5**: stays small and of varying sign (a few %), already resolved, no floor.

**Cautious verdict (l=4, the question posed).** To the question "does the l=4 O5 residual keep
closing toward 0 % (-> it was diffusion, not a hard floor) or does it plateau at a given % (-> a floor
of that size)?", the high-resolution measurement answers: **it shows no closure toward 0 % in these
measurements**. PR-0's -4 % was an artifact of the only two clean low-resolution points; at n=384 and
n=512 l=4 O5 lands at ~-9.5 %, the same order as the l=3 plateau (~-9 %) which is itself measured
cleanly and flat over one step (n=384 -> n=512). The reading most consistent with the full dataset is
therefore that the data suggest an l-dependent residual floor on the order of **~9-10 %** at order 5,
which shows no closure toward 0 with resolution in these measurements: the structural candidate of
the cartesian ring boundary targeted by PR-A "transport-wall", not yet definitive proof. This reading
re-opens PR-0's floor hypothesis (which the low-resolution O5 axis had seemed to weaken), placing its
plausible size around ~9-10 % at order 5 (against ~12 % seen at O2), subject to the two limits below.

This verdict remains to be confirmed and on its own justifies no rewrite of the paper roadmap:
(1) the l=3 plateau so far holds only over one flat step (n=384 -> n=512); you would need either an
n=768/1024 or two different `t_end` horizons to rule out a merely very slow convergence; (2) the l=4
high-resolution points are biased by their early fit window, before putting a number on an l=4 floor,
you need a robust window diagnostic (open the fit later, or anchor the window on the clean exponential
phase as at points n=128/256). In other words, the "diffusion vs structural" question now leans, on
the basis of this measurement, toward the **structural** side **(floor ~9-10 % at order 5)** rather
than diffusion, but this shift relative to PR-0's cautious verdict is only suggested, and remains to
be confirmed.

## What was run vs skipped

- **Run (main O1/O2 grid, 27 runs)**: `n in {128,192,256} x {O1 none, O2 minmod, O2 vanleer} x
  l in {3,4,5}`, `t_end=48`. Replayed here identically to PR-0 (same values), so the O1/O2 rows of
  the table are stable. Replayable with `sweep.py --orders none,minmod,vanleer`.
- **Run (high-order axis O5, 9 runs)**: `n in {128,192,256} x O5 (weno5 + ssprk3) x l in {3,4,5}`,
  `t_end=48`. Per-run costs measured locally (single-thread CPU): n=128 ~ 9 s, n=192 ~ 30 s,
  n=256 ~ 88 s (WENO5-Z = 5-point stencil + 3 ghosts, SSPRK3 = 3 stages, so heavier than O2 at the
  same n). The full 4-order x 3-n x 3-l sweep = **36 runs in ~16.5 min** (local CPU). Complete CSV,
  replayable with `sweep.py` (default, which now includes `weno5`).
- **Run (n=384 O2 probe, 6 runs)**: `n=384 x {O2 minmod, O2 vanleer} x l in {3,4,5}` (inherited from
  PR-0; one n=384 O2 run ~ 143 s locally; replayable via `sweep.py --ns 384 --orders minmod,vanleer`).
  O1 skipped at n=384 (order 1 stays diffusion-dominated at any resolution).
- **Run on ROMEO (high resolution n=384/512, 18 runs)**: `n in {384, 512} x {O2 minmod, O2 vanleer,
  O5 weno5} x l in {3,4,5}`, `t_end=48`, SLURM job **639912** (partition `short`,
  `--constraint=x64cpu`, amd EPYC 192 cores, account `r250127`). The 18 runs (each single-thread) run
  in parallel on one node; all rc=0, no NaN, `t_final = 48` reached without hitting the `max_steps=4000`
  guard (n=512 needs ~2600 steps, so a comfortable margin). The `_adc` module is rebuilt on the ROMEO
  login node (Spack: python@3.10.14 + numpy@1.26.4 + pybind11@2.13.5 + cmake@3.31.8 + gcc@11.4.1) from
  `adc_cpp` master `5bb7208`; **the n=384 and n=512 O2 rows reproduce the n=384 O2 values of PR-0 to
  the hundredth**, which validates the ROMEO chain. Wall-time per run measured (EPYC CPU, 18 then 9
  then 3 concurrent runs -> bandwidth contention at n=512):
  - n=384 O2 (minmod/vanleer): ~172-182 s; n=384 O5 (weno5+ssprk3): ~249-258 s.
  - n=512 O2 (minmod/vanleer): ~653-716 s; n=512 O5: ~833-880 s (heaviest point: n=512 O5 l=4 =
    880 s).
  - Full job (18 runs in parallel): ~14.5 min wall. Replayable with `sweep.py --ns 384,512
    --orders minmod,vanleer,weno5`.
- **Still skipped**: **n=384/512 O1 none** (order 1 stays diffusion-dominated at any resolution, of no
  interest for the diffusion-vs-structural question). Avenues opened by this measurement, not done here
  (see verdict): (a) **n=768/1024** or **two `t_end` horizons** to rule out that the l=3 O5 plateau
  ~ -9 % is a very slow convergence; (b) a **robust window diagnostic for l=4** (fit window anchored
  late, as at the clean points n=128/256) before putting a number on an l=4 floor: the l=4
  high-resolution points have an early window that under-reads the slope.

## Reproduce

```bash
# adc_cpp est Kokkos-only : un Kokkos installe (Serial pour CPU) est requis (-DKokkos_ROOT).
cd ../adc_cpp && cmake -S . -B build-py -DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=ON \
  -DKokkos_ROOT=$KOKKOS_ROOT -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build-py -j4
cd ../adc_cases
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py        # O2 + O5 (defaut)
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --orders weno5  # O5 seul
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py --quick # fumee rapide
PYTHONPATH=../adc_cpp/build-py/python python3 diocotron/sweep.py \
  --ns 384,512 --orders minmod,vanleer,weno5         # haute resolution (lourd : ROMEO)
```

The `--orders` default is now `minmod,vanleer,weno5` (O5 = WENO5-Z + SSPRK3). Add `none` for the O1
row. The O5 axis requires adc_cpp #88 or newer (master `ca803dc`). `sweep.py` was not modified for
high resolution: it already parameterizes `--ns` and `--orders`, and the `max_steps=4000` guard covers
n=512 (~2600 steps at `t_end=48`).

On ROMEO (high resolution, job 639912 above): the sweep is CPU (Poisson + transport, no GPU),
partition `short` `--constraint=x64cpu`, and the `_adc` module is rebuilt on the login node from
`adc_cpp` (Spack: python@3.10.14 + numpy@1.26.4 + pybind11@2.13.5 + cmake@3.31.8 + gcc@11.4.1, plus an
installed Kokkos) via the same command `cmake -S . -B build-py -DADC_BUILD_PYTHON=ON
-DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT -DCMAKE_BUILD_TYPE=Release` (adc_cpp is Kokkos-only: the
serial path goes through Kokkos Serial). The 18 runs (n=384/512 x 3 orders x 3 modes) run single-thread
in parallel on one node (192 cores).
