% golden_init_gen.m -- ADC-350
%
% Dump the initial conditions of the HyQMOM15 cases of the new reference
% RieMOM2D_Electrostatic_periodic, at a small grid (Np=16) so the CSVs stay
% tiny while still exercising the full recipe (eigenmode + phase + Maxwellian
% for the waves, ring + periodic FFT Poisson + ExB drift + Maxwellian for the
% diocotron). The IC recipe is local for the waves/constant and the diocotron
% pipeline is Np-parametrized, so Np=16 parity validates the implementation;
% the per-driver full-Np fields follow in the driver PRs (ADC-351+).
%
% Run from the adc_cases repo root:
%   octave --no-gui -p /Users/romaindespoulain/Documents/RieMOM2D_Electrostatic_periodic \
%       hyqmom15/matlab_ref/golden_init_gen.m
%
% Variants dumped: the intended IC of every case, plus magnetic_wave_aswritten
% (the D4 oversight: electrostatic wiring) and dicotron_matlab_bug (the D2
% transposed meshgrid drift). Fields are block-stacked (15*Np x Np), %.17g.

pkg_dir = fileparts(mfilename('fullpath'));
gdir = fullfile(pkg_dir, 'goldens');
if ~exist(gdir, 'dir'); mkdir(gdir); end

Np = 16;          % golden resolution (documented; recipe is resolution-faithful)
L = 1.0;
dx = L / Np;
Mi = InitializeM4_15(1, 0, 0, 1, 0, 1);   % equilibrium Maxwellian, [1 0 1 0 3 ...]

function vec = phase_pin(vec)
  vec = vec / norm(vec);
  [~, k] = max(abs(vec));
  vec = vec * conj(vec(k)) / abs(vec(k));
end

function ind = canon_sort(w)
  % Identical to linearized.matlab_sort_indices (see golden_linearized_gen.m):
  % deterministic, BLAS-robust mode ordering.
  w = w(:);
  tol = 1e-9;
  scale = max(1.0, max(abs(w)));
  if max(abs(imag(w))) <= tol * scale
    [~, ind] = sort(real(w));
  else
    q = round(abs(w) / scale * 1e9) / 1e9;
    [~, ind] = sortrows([q, real(w), imag(w)]);
  end
end

function M = wave_ic(J, kx, ky, ep, mode, Mi, Np, dx)
  [V, D] = eig(J);
  ind = canon_sort(diag(D));
  Vs = V(:, ind);
  rv = real(phase_pin(Vs(:, mode)));     % realified, phase-pinned eigenvector
  M = zeros(Np, Np, 15);
  for i = 1:Np
    for j = 1:Np
      phase = kx * (i - 1) * dx + ky * (j - 1) * dx;
      for k = 1:15
        M(i, j, k) = Mi(k) + ep * rv(k) * sin(phase);
      end
    end
  end
end

function M = dioc_ic(orientation, Np, dx, r0, r1, rmin, rmax, ep, mode, debye, oc)
  xm = -0.5 + ((1:Np) - 0.5) * dx;
  [X, Y] = meshgrid(xm, xm);
  R = sqrt(X .^ 2 + Y .^ 2);
  th = atan2(Y, X);
  rho = rmin * ones(Np, Np);
  mask = (R >= r0) & (R <= r1);
  delta = 1 - ep + ep * sin(mode * th);
  rho(mask) = rmax * delta(mask);
  phi = poisson_fft(rho, debye, dx, dx);
  g1 = (circshift(phi, -1, 1) - circshift(phi, 1, 1)) / (2 * dx);  % d/d(first index)
  g2 = (circshift(phi, -1, 2) - circshift(phi, 1, 2)) / (2 * dx);  % d/d(second index)
  if strcmp(orientation, 'matlab_bug')
    vx = -g2 / oc;  vy = g1 / oc;   % transposed/divergent (init_diocotron_field.m)
  else
    vx = -g1 / oc;  vy = g2 / oc;   % corrected incompressible ExB
  end
  M = zeros(Np, Np, 15);
  M(:, :, 1) = rho;
  M(:, :, 2) = rho .* vx;            M(:, :, 6) = rho .* vy;
  M(:, :, 3) = rho .* (vx .^ 2 + 1); M(:, :, 7) = rho .* (vx .* vy);  M(:, :, 10) = rho .* (vy .^ 2 + 1);
  M(:, :, 4) = rho .* (vx .^ 3 + 3 * vx);
  M(:, :, 8) = rho .* ((vx .^ 2 + 1) .* vy);
  M(:, :, 11) = rho .* ((vy .^ 2 + 1) .* vx);
  M(:, :, 13) = rho .* (vy .^ 3 + 3 * vy);
  M(:, :, 5) = rho .* (vx .^ 4 + 6 * vx .^ 2 + 3);
  M(:, :, 9) = rho .* (vx .^ 3 .* vy + 3 * vx .* vy);
  M(:, :, 12) = rho .* (vx .^ 2 .* vy .^ 2 + (vx .^ 2 + vy .^ 2) + 1);
  M(:, :, 14) = rho .* (vx .* vy .^ 3 + 3 * vx .* vy);
  M(:, :, 15) = rho .* (vy .^ 4 + 6 * vy .^ 2 + 3);
end

function dump_field(gdir, tag, M, Np)
  out = zeros(15 * Np, Np);
  for k = 1:15
    out((k - 1) * Np + 1:k * Np, :) = M(:, :, k);
  end
  dlmwrite(fullfile(gdir, ['init_' tag '.csv']), out, 'precision', '%.17g');
end

% Wave cases (intended).
dump_field(gdir, 'fluid_wave', ...
           wave_ic(linearized_Jacobian_fluid(4 * pi / L, 0 * pi / L), 4 * pi / L, 0 * pi / L, 0.01, 15, Mi, Np, dx), Np);
dump_field(gdir, 'electrostatic_wave', ...
           wave_ic(linearized_Jacobian_electrostatic(0 * pi / L, 4 * pi / L, 1 / 30), 0 * pi / L, 4 * pi / L, 0.01, 15, Mi, Np, dx), Np);
dump_field(gdir, 'magnetic_wave', ...
           wave_ic(linearized_Jacobian_magnetostatic(2 * pi / L, 4 * pi / L, 1 / 20, -40), 2 * pi / L, 4 * pi / L, 0.01, 15, Mi, Np, dx), Np);
% magnetic_wave as_written (D4 oversight: electrostatic wiring, magnetic kx/ky/debye).
dump_field(gdir, 'magnetic_wave_aswritten', ...
           wave_ic(linearized_Jacobian_electrostatic(2 * pi / L, 4 * pi / L, 1 / 20), 2 * pi / L, 4 * pi / L, 0.01, 15, Mi, Np, dx), Np);

% Diocotron (standard corrected ExB, and the matlab_bug transposed drift).
dump_field(gdir, 'dicotron_standard', ...
           dioc_ic('standard', Np, dx, 0.35, 0.4, 1e-4, 1.0, 0.1, 4, 1 / 20, -20), Np);
dump_field(gdir, 'dicotron_matlab_bug', ...
           dioc_ic('matlab_bug', Np, dx, 0.35, 0.4, 1e-4, 1.0, 0.1, 4, 1 / 20, -20), Np);

% Constant (uniform equilibrium).
Mc = zeros(Np, Np, 15);
for k = 1:15; Mc(:, :, k) = Mi(k); end
dump_field(gdir, 'constant', Mc, Np);

printf('init goldens (Np=%d) written to %s\n', Np, gdir);
