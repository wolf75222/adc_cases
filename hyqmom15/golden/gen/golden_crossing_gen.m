% golden/gen/golden_crossing_gen.m : produit le golden de parite des conditions initiales correlees,
% en faisant tourner InitializeM4_15 (RIEMOM2D) sur des etats gaussiens (M00, u, v, C20, C11, C02).
%
% But : montrer que InitializeM4_15 (S22=1, S31=S13=0 dans la base principale + rotation S4toC4
% par C11) donne EXACTEMENT les 15 moments bruts d'une gaussienne correlee de covariance
% [[C20, C11], [C11, C02]] -- ceux que gaussian_state (formule d'Isserlis) calcule cote Python.
%
% Sortie (dossier golden/) :
%   golden_crossing.csv : une ligne par etat, colonnes
%     [M00, u, v, C20, C11, C02, M(1)..M(15)]  (6 parametres + 15 moments).
%
% Usage (depuis hyqmom15/) :
%   octave --no-gui --path /chemin/vers/RIEMOM2D golden/gen/golden_crossing_gen.m
%
% Provenance a consigner dans le README : version d'Octave, SHA du depot RIEMOM2D.

% Parametres : (M00, u, v, C20, C11, C02). C11 = r*sqrt(C20*C02).
% On couvre r = 0, 0.5, -0.5 (cas isotrope C20=C02=T=1) plus un cas anisotrope C20 != C02.
T = 1.0;
params = [ ...
  % isotrope C20=C02=1, r balaye : repos puis jets du croisement (Uc = Ma/sqrt(2), Ma=20)
  1.0,      0.0,      0.0, T, 0.0*sqrt(T*T),  T;   % r=0,    repos
  1.0,      0.0,      0.0, T, 0.5*sqrt(T*T),  T;   % r=0.5,  repos
  1.0,      0.0,      0.0, T,-0.5*sqrt(T*T),  T;   % r=-0.5, repos
  1.0, -14.1421356237309515, -14.1421356237309515, T, 0.5*sqrt(T*T), T;  % r=0.5, jet (-Uc,-Uc)
  1.0,  14.1421356237309515,  14.1421356237309515, T,-0.5*sqrt(T*T), T;  % r=-0.5, jet (+Uc,+Uc)
  % anisotrope C20 != C02, r=0.5 : C11 = r*sqrt(C20*C02)
  2.0,      0.5,     -0.3, 1.5, 0.5*sqrt(1.5*0.7), 0.7];

N = size(params, 1);
OUT = zeros(N, 6 + 15);
for i = 1:N
  M00 = params(i, 1);
  u   = params(i, 2);
  v   = params(i, 3);
  C20 = params(i, 4);
  C11 = params(i, 5);
  C02 = params(i, 6);
  M = InitializeM4_15(M00, u, v, C20, C11, C02);
  OUT(i, 1:6) = params(i, :);
  OUT(i, 7:21) = M(:)';
end

dlmwrite(fullfile("golden", "golden_crossing.csv"), OUT, "precision", "%.17g");
printf("golden de parite ecrit pour %d etats correles (6 parametres + 15 moments)\n", N);
