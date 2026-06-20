% golden_linearized_gen.m -- ADC-350
%
% Dump the linearized Jacobians, the Matlab-sorted eigenvalues, and the
% phase-pinned mode-15 eigenvector for the HyQMOM15 wave cases of the new
% reference RieMOM2D_Electrostatic_periodic, plus the CFL max_speed values.
% These goldens lock the linear-algebra reference that hyqmom15/matlab_ref/
% linearized.py is checked against (check_goldens.py).
%
% Run from the adc_cases repo root (the -p path is the maintainer-side Matlab):
%   octave --no-gui -p /Users/romaindespoulain/Documents/RieMOM2D_Electrostatic_periodic \
%       hyqmom15/matlab_ref/golden_linearized_gen.m
%
% Case parameters mirror hyqmom15/matlab_ref/params.py (= the live init_*.m).
% Domain side L = xmax - xmin = 1. All CSVs are %.17g, no header.

pkg_dir = fileparts(mfilename('fullpath'));
gdir = fullfile(pkg_dir, 'goldens');
if ~exist(gdir, 'dir'); mkdir(gdir); end
L = 1.0;

function vec = phase_pin(vec)
  % Same gauge fix as linearized.phase_pin: L2-normalize, then rotate so the
  % largest-magnitude component is real and positive.
  vec = vec / norm(vec);
  [~, k] = max(abs(vec));
  vec = vec * conj(vec(k)) / abs(vec(k));
end

function ind = canon_sort(w)
  % Identical to linearized.matlab_sort_indices: real spectra ascending by value,
  % complex spectra ascending by magnitude with near-magnitude ties broken by
  % (real, imag). Robust to the ~1e-13 Octave/NumPy LAPACK difference.
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

function s = canon_last(w)
  ind = canon_sort(w);
  s = w(ind)(end);
end

function dump_case(gdir, tag, J, mode)
  [V, D] = eig(J);
  lam = diag(D);
  ind = canon_sort(lam);          % deterministic, BLAS-robust (matches the layer)
  ls = lam(ind);
  Vs = V(:, ind);
  vec = phase_pin(Vs(:, mode));   % mode is 1-based
  dlmwrite(fullfile(gdir, ['lin_' tag '_jac.csv']), [real(J) imag(J)], 'precision', '%.17g');
  dlmwrite(fullfile(gdir, ['lin_' tag '_eigvals.csv']), [real(ls) imag(ls)], 'precision', '%.17g');
  dlmwrite(fullfile(gdir, ['lin_' tag '_eigvec.csv']), [real(vec) imag(vec)], 'precision', '%.17g');
end

% Wave cases: Jacobian + eigenmode (intended wiring).
dump_case(gdir, 'fluid_wave',         linearized_Jacobian_fluid(4 * pi / L, 0 * pi / L), 15);
dump_case(gdir, 'electrostatic_wave', linearized_Jacobian_electrostatic(0 * pi / L, 4 * pi / L, 1 / 30), 15);
dump_case(gdir, 'magnetic_wave',      linearized_Jacobian_magnetostatic(2 * pi / L, 4 * pi / L, 1 / 20, -40), 15);

% CFL max_speed values. es intended = diag(Dmax) at (kmin,kmin), kmin=2*pi/dx,
% dx=1/128; es as_written = diag(D) at the mode (kx=0,ky=4*pi); diocotron =
% magnetostatic at (kmin,kmin), kmin=sqrt(2)*pi, real-part sort.
dx_es = 1 / 128;
es_intended  = canon_last(eig(linearized_Jacobian_electrostatic(2 * pi / dx_es, 2 * pi / dx_es, 1 / 30)));
es_aswritten = canon_last(eig(linearized_Jacobian_electrostatic(0 * pi / L, 4 * pi / L, 1 / 30)));
lamd = sort(real(eig(linearized_Jacobian_magnetostatic(sqrt(2) * pi / L, sqrt(2) * pi / L, 1 / 20, -20))));
dioc_ms = lamd(end);
dlmwrite(fullfile(gdir, 'maxspeed.csv'), ...
         [real(es_intended) imag(es_intended); ...
          real(es_aswritten) imag(es_aswritten); ...
          dioc_ms 0], 'precision', '%.17g');

printf('linearized goldens written to %s\n', gdir);
