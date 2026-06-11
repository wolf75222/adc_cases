% golden_hll_gen.m : mini-crossing HLL de REFERENCE, schema MATLAB fidele (Octave, RIEMOM2D).
%
% Produit le golden de fidelite HLL d'ADC-89 : croisement de jets a Ma = 2 (Np = 64, 20 pas,
% sans relaxation), avance par le SCHEMA DU DEPOT DE REFERENCE -- vitesses par
% eigenvalues15_2D(M, 1) (jacobienne symbolique + eig par blocs), flux par Flux_closure15_2D,
% HLL de Davis par pas_HLL, split dimensionnel ADDITIF Mnp = Mnpx + Mnpy - M, Euler explicite.
% Convention de grille de main_electrostatic_wave.m : interieur Np x Np + UNE rangee de ghosts
% periodiques (M_ext de taille Np+2), rafraichis avant chaque balayage.
%
% La SEQUENCE DE dt est enregistree : le comparateur adc (run_crossing.py, check HLL) REJOUE
% exactement ces dt (sim.step(dt)) pour eliminer la difference d'horloge -- l'ecart residuel
% mesure alors la difference de SCHEMA (split additif Euler vs non-splite ssprk2), documentee.
%
% Sorties (golden/) : golden_hll_state.csv (etat final interieur, (15*Np) x Np, bloc par
% moment), golden_hll_dts.csv (dt par pas), golden_hll_meta.csv (Np, Ma, nsteps, CFL).
%
% Usage : octave --no-gui --path /chemin/vers/RIEMOM2D golden_hll_gen.m

Np = 64; CFL = 0.4; Ma = 2.0; T = 1.0; rhol = 1.0; rhor = 1e-3; nsteps = 20;
xmin = -0.5; xmax = 0.5;
dx = (xmax - xmin) / Np; dy = dx;
Nmom = 15;

% --- IC croisement (main_pb_2Dcrossing_2DHyQMOM15.m), sur l'INTERIEUR ---
C20 = T; C02 = T; C11 = 0.0;
Ml = InitializeM4_15(rhol, 0, 0, C20, C11, C02);
Mr = InitializeM4_15(rhor, 0, 0, C20, C11, C02);
Uc = Ma / sqrt(2);
Mt = InitializeM4_15(rhol, -Uc, -Uc, C20, C11, C02);
Mb = InitializeM4_15(rhol,  Uc,  Uc, C20, C11, C02);
M = zeros(Np, Np, Nmom);
for i = 1:Np
  for j = 1:Np
    M(i, j, :) = Mr;
  end
end
for i = 3*Np/8+1:5*Np/8
  for j = 3*Np/8+1:5*Np/8
    if i + j == Np + 1
      M(i, j, :) = Ml;
    elseif i + j > Np
      M(i, j, :) = Mt;
    else
      M(i, j, :) = Mb;
    end
  end
end

% --- boucle temporelle : schema de main_electrostatic_wave.m (M_ext + ghosts periodiques) ---
dts = zeros(nsteps, 1);
M_ext = zeros(Np+2, Np+2, Nmom);
Fx = zeros(Np+2, Np+2, Nmom); Fy = Fx;
vpxmin = zeros(Np+2, Np+2); vpxmax = vpxmin; vpymin = vpxmin; vpymax = vpxmin;

for nn = 1:nsteps
  % ghosts periodiques
  M_ext(2:Np+1, 2:Np+1, :) = M;
  M_ext(1, 2:Np+1, :)    = M(Np, :, :);
  M_ext(Np+2, 2:Np+1, :) = M(1, :, :);
  M_ext(2:Np+1, 1, :)    = M(:, Np, :);
  M_ext(2:Np+1, Np+2, :) = M(:, 1, :);
  % coins (lus par la boucle vp/flux ci-dessous, pas par les balayages) : enroulement periodique
  M_ext(1, 1, :)       = M(Np, Np, :);
  M_ext(1, Np+2, :)    = M(Np, 1, :);
  M_ext(Np+2, 1, :)    = M(1, Np, :);
  M_ext(Np+2, Np+2, :) = M(1, 1, :);

  % vitesses + flux partout (ghosts compris : pas_HLL lit les bords)
  for i = 1:Np+2
    for j = 1:Np+2
      MOM = squeeze(M_ext(i, j, :));
      [vpxmin(i,j), vpxmax(i,j), vpymin(i,j), vpymax(i,j)] = eigenvalues15_2D(MOM, 1);
      [mx, my] = Flux_closure15_2D(MOM);
      Fx(i, j, :) = mx;
      Fy(i, j, :) = my;
    end
  end

  dt = CFL * dx / max([abs(vpxmax); abs(vpxmin); abs(vpymax); abs(vpymin)], [], 'all');
  dts(nn) = dt;

  % balayage x (par ligne j) puis y (par colonne i), split additif
  Mnpx = M_ext;
  for j = 2:Np+1
    MNP = pas_HLL(squeeze(M_ext(:, j, :)), squeeze(Fx(:, j, :)), dt, dx, ...
                  vpxmin(:, j), vpxmax(:, j));
    Mnpx(:, j, :) = MNP;
  end
  Mnpy = M_ext;
  for i = 2:Np+1
    MNP = pas_HLL(squeeze(M_ext(i, :, :)), squeeze(Fy(i, :, :)), dt, dy, ...
                  vpymin(i, :)', vpymax(i, :)');
    Mnpy(i, :, :) = MNP;
  end
  Mnp = Mnpx + Mnpy - M_ext;
  M = Mnp(2:Np+1, 2:Np+1, :);
end

% --- sorties ---
out = zeros(Nmom * Np, Np);
for k = 1:Nmom
  out((k-1)*Np+1 : k*Np, :) = M(:, :, k);
end
dlmwrite(fullfile("golden", "golden_hll_state.csv"), out, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_hll_dts.csv"), dts, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_hll_meta.csv"), [Np, Ma, nsteps, CFL], "precision", "%.17g");
printf("golden HLL ecrit : Np=%d, Ma=%g, %d pas, t_final=%.6g\n", Np, Ma, nsteps, sum(dts));
