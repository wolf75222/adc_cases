% golden/gen/golden_crossing_relax_gen.m : mini-crossing HLL Ma=20 AVEC relaxation15 active (flagrelax=1),
% schema MATLAB fidele (Octave, RIEMOM2D).
%
% Comble le trou de couverture transport x relaxation : golden/gen/golden_hll_gen.m tourne flagrelax=0 ET
% Ma=2 (regime ou relaxation15 ne se declenche jamais) ; golden/gen/golden_relax_gen.m couvre la projection
% ISOLEE. AUCUN golden n'enchaine transport puis relaxation comme le pilote de production
% (main_pb_2Dcrossing_2DHyQMOM15.m:37, flagrelax=1, Ma=20), qui applique relaxation15 a chaque
% cellule a chaque pas APRES le split additif (lignes 278-291 du pilote).
%
% On reprend les briques REELLES de golden/gen/golden_hll_gen.m (vitesses eigenvalues15_2D(M, 1), flux
% Flux_closure15_2D, HLL de Davis pas_HLL, split dimensionnel ADDITIF Mnp = Mnpx + Mnpy - M,
% Euler explicite) et on AJOUTE, comme le pilote, l'etape flagrelax=1 : relaxation15(MM, lamin,
% Ma) par cellule sur Mnp avant de poser M = Mnp. La collision BGK du pilote est inactive (Kn =
% 1000 > 10 dans le pilote), non portee.
%
% La SEQUENCE DE dt est enregistree : le comparateur adc (run_relaxation.py, check de replay)
% REJOUE exactement ces dt cote Python (transport euler par sim.step(dt) PUIS relax_field) pour
% eliminer la difference d'horloge -- l'ecart residuel mesure la fidelite du port (flux/vitesses
% identiques, relaxation15 transcrite branche par branche).
%
% Sorties (golden/) : golden_crossing_relax_out.csv (etat final interieur, (15*Np) x Np, bloc par
% moment), golden_crossing_relax_in.csv (IC interieure, meme disposition, pour rejouer la MEME IC
% cote Python sans dependre de crossing_state), golden_crossing_relax_meta.csv
% (Np, Ma, nsteps, CFL, lamin) ; golden_crossing_relax_dts.csv (dt par pas).
%
% Usage : octave --no-gui --path /chemin/vers/RIEMOM2D golden/gen/golden_crossing_relax_gen.m

Np = 32; CFL = 0.5; Ma = 20.0; T = 1.0; rhol = 1.0; rhor = 1e-3; nsteps = 3;
lamin = 1.d-12;
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

% etat initial interieur sauvegarde (pour rejouer la MEME IC cote Python)
in_out = zeros(Nmom * Np, Np);
for k = 1:Nmom
  in_out((k-1)*Np+1 : k*Np, :) = M(:, :, k);
end

% --- boucle temporelle : schema de golden/gen/golden_hll_gen.m (M_ext + ghosts periodiques) + flagrelax=1 ---
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

  % flagrelax = 1 : relaxation15 par cellule sur l'INTERIEUR (comme le pilote, lignes 278-291).
  % Applique APRES le split additif, sur Mnp, avant de poser M = Mnp.
  Mint = Mnp(2:Np+1, 2:Np+1, :);
  for i = 1:Np
    for j = 1:Np
      MM = squeeze(Mint(i, j, :));
      MMC = relaxation15(MM, lamin, Ma);
      Mint(i, j, :) = MMC;
    end
  end
  M = Mint;
end

% --- sorties ---
out = zeros(Nmom * Np, Np);
for k = 1:Nmom
  out((k-1)*Np+1 : k*Np, :) = M(:, :, k);
end
mkdir('golden');
dlmwrite(fullfile("golden", "golden_crossing_relax_in.csv"),  in_out, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_crossing_relax_out.csv"), out,    "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_crossing_relax_dts.csv"), dts,    "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_crossing_relax_meta.csv"), ...
         [Np, Ma, nsteps, CFL, lamin], "precision", "%.17g");
printf("golden crossing+relax ecrit : Np=%d, Ma=%g, %d pas, flagrelax=1, t_final=%.6g\n", ...
       Np, Ma, nsteps, sum(dts));
