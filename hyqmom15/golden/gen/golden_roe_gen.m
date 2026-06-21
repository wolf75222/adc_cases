% golden/gen/golden_roe_gen.m : one-step ROE golden of REFERENCE (Octave, RieMOM2D_Electrostatic_periodic).
%
% Produces the strict-parity golden for hyqmom15/fluid_wave (ADC-380): ONE ROE + Euler step of
% the reference scheme on the fluid_wave eigenmode IC (Np=32, eps=0.01, mode=15, kx=4*pi, ky=0),
% advanced by the reference's OWN functions -- spatial_operator (space_scheme="ROE" -> flux_ROE
% -> flux_ROE_local: arithmetic Roe average Uavg=1/2(UL+UR), A=jacobian15_2D(Uavg), eig-based
% |A| with Harten fix, then compute_div = -(dFdx+dFdy)) + euler_step (M1 = M0 + dt*dU).
% fluid_wave has source=0, so phi is inert.
%
% The SAME dt is recorded; run_fluid_wave.py (check_golden_roe) seeds the EXACT dumped IC and
% replays this dt (sim.step(dt)) with riemann="roe" -- the residual then measures the ROE
% OPERATOR difference alone (closure + flux Jacobian + matrix-sign |A| + divergence), at the
% eigendecomposition / floating-point tolerance.
%
% Outputs (golden/): golden_roe_state0.csv (IC M^0) and golden_roe_state.csv (M^1), both
% (15*Np) x Np block-per-moment (same layout as golden_hll_state.csv); golden_roe_dt.csv (the
% single dt); golden_roe_meta.csv (Np, CFL).
%
% Usage (run from hyqmom15/, the outputs go to ./golden/):
%   octave --no-gui --path /path/to/RieMOM2D_Electrostatic_periodic golden/gen/golden_roe_gen.m

params = init_fluid_wave(struct());
Np = params.Np; Nmom = params.Nmom; dx = params.dx; CFL = params.CFL;

% --- IC eigenmode (reference init_fluid_wave_field) ---
[M0, params] = init_fluid_wave_field(params, []);
if max(abs(imag(M0(:)))) > 1e-13
  error("golden_roe_gen: IC has non-negligible imaginary part (%.3e)", max(abs(imag(M0(:)))));
end
M0 = real(M0);

% --- dt = CFL*dx/vmax (reference compute_dt; fluid_wave source=0 -> no source cap) ---
% vmax over interior + one ring of periodic ghosts, eigenvalues15_2D (same as the HLL golden).
M_ext = zeros(Np+2, Np+2, Nmom);
M_ext(2:Np+1, 2:Np+1, :) = M0;
M_ext(1, 2:Np+1, :)    = M0(Np, :, :);
M_ext(Np+2, 2:Np+1, :) = M0(1, :, :);
M_ext(2:Np+1, 1, :)    = M0(:, Np, :);
M_ext(2:Np+1, Np+2, :) = M0(:, 1, :);
M_ext(1, 1, :)       = M0(Np, Np, :);
M_ext(1, Np+2, :)    = M0(Np, 1, :);
M_ext(Np+2, 1, :)    = M0(1, Np, :);
M_ext(Np+2, Np+2, :) = M0(1, 1, :);
vmax = 0;
for i = 1:Np+2
  for j = 1:Np+2
    MOM = squeeze(M_ext(i, j, :));
    [vpxmin, vpxmax, vpymin, vpymax] = eigenvalues15_2D(MOM, params.flagsym);
    vmax = max([vmax, abs(vpxmin), abs(vpxmax), abs(vpymin), abs(vpymax)]);
  end
end
dt = compute_dt(vmax, params, 0);

% --- one ROE + Euler step (reference spatial_operator + euler_step) ---
phi = zeros(Np, Np);
rhs = @(U) spatial_operator(U, phi, params);
M1 = euler_step(M0, rhs, dt);
if max(abs(imag(M1(:)))) > 1e-11
  error("golden_roe_gen: M1 has non-negligible imaginary part (%.3e)", max(abs(imag(M1(:)))));
end
M1 = real(M1);

% --- outputs (block-per-moment, (15*Np) x Np : block k = M(:,:,k), same as golden_hll_gen) ---
out0 = zeros(Nmom * Np, Np);
out1 = zeros(Nmom * Np, Np);
for k = 1:Nmom
  out0((k-1)*Np+1 : k*Np, :) = M0(:, :, k);
  out1((k-1)*Np+1 : k*Np, :) = M1(:, :, k);
end
dlmwrite(fullfile("golden", "golden_roe_state0.csv"), out0, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_roe_state.csv"),  out1, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_roe_dt.csv"), dt, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_roe_meta.csv"), [Np, CFL], "precision", "%.17g");
printf("golden ROE one-step written: Np=%d, dt=%.6g, |M1-M0|_inf=%.3e\n", ...
       Np, dt, max(abs(M1(:) - M0(:))));
