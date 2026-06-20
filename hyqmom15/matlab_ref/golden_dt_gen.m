% golden_dt_gen.m -- ADC-350
%
% Dump the time-step policy by calling the actual compute_dt.m of the new
% reference RieMOM2D_Electrostatic_periodic, so hyqmom15/matlab_ref/dt_policy.py
% is cross-validated against the Matlab oracle (not a re-transcription). Each
% row is [case_index, vmax, t, dt]; the case order matches CASE_NAMES below.
%
% Run from the adc_cases repo root:
%   octave --no-gui -p /Users/romaindespoulain/Documents/RieMOM2D_Electrostatic_periodic \
%       hyqmom15/matlab_ref/golden_dt_gen.m
%
% Probe points exercise: the bare CFL bound, the electrostatic and (would-be)
% magnetostatic source caps, the both-source elseif, and the final-time clamp.

pkg_dir = fileparts(mfilename('fullpath'));
gdir = fullfile(pkg_dir, 'goldens');
if ~exist(gdir, 'dir'); mkdir(gdir); end

function p = mk(cfl, dx, tmax, es, ms, op, oc)
  p = struct('CFL', cfl, 'dx', dx, 'tmax', tmax, ...
             'electrostatic', es, 'magnetostatic', ms, 'omega_p', op, 'omega_c', oc);
  p.case_name = 'probe';
end

% case_index -> params (mirrors params.py / init_*.m).
cases = { mk(0.4, 1 / 32,  0.05, 0, 0, 30, -90), ...   % 1 fluid_wave
          mk(0.5, 1 / 128, 1.0,  1, 0, 30, -90), ...   % 2 electrostatic_wave
          mk(0.5, 1 / 256, 1.0,  1, 1, 20, -40), ...   % 3 magnetic_wave
          mk(0.5, 1 / 128, 1.0,  1, 1, 20, -20), ...   % 4 dicotron
          mk(0.5, 1 / 64,  1.0,  0, 0, 30, -90) };      % 5 constant

rows = [];
for ci = 1:numel(cases)
  p = cases{ci};
  for vmax = [10.0, 200.0]
    for t = [0.0, p.tmax - 1e-6]
      dt = compute_dt(vmax, p, t);
      rows = [rows; ci, vmax, t, dt];
    end
  end
end
dlmwrite(fullfile(gdir, 'dt.csv'), rows, 'precision', '%.17g');
printf('dt golden written to %s (%d rows)\n', gdir, size(rows, 1));
