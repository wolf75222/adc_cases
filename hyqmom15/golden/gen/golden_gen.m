% golden/gen/golden_gen.m : produit les goldens HyQMOM en faisant tourner le VRAI code MATLAB de reference
% (RIEMOM2D) sur les etats de golden/golden_states.csv.
%
% Sorties (memes dossier golden/) :
%   golden_fx.csv, golden_fy.csv : flux physiques Flux_closure15_2D(M) par etat (N x 15) ;
%   golden_vp.csv : [vpxmin vpxmax vpymin vpymax] de eigenvalues15_2D(M, 1) par etat (N x 4)
%                   -- chemin flagsym=1 (jacobian15 symbolique + eig par blocs), celui de la
%                   production MATLAB ; reference future d'.
%
% Usage (depuis hyqmom15/) :
%   octave --no-gui --path /chemin/vers/RIEMOM2D golden/gen/golden_gen.m
%
% Provenance a consigner dans le README : version d'Octave, SHA du depot RIEMOM2D.

states = dlmread(fullfile("golden", "golden_states.csv"));
N = size(states, 1);
assert(size(states, 2) == 15);

FX = zeros(N, 15);
FY = zeros(N, 15);
VP = zeros(N, 4);
for i = 1:N
  M = states(i, :);
  [fx, fy] = Flux_closure15_2D(M);
  FX(i, :) = fx;
  FY(i, :) = fy;
  [vpxmin, vpxmax, vpymin, vpymax] = eigenvalues15_2D(M(:), 1);
  VP(i, :) = [vpxmin, vpxmax, vpymin, vpymax];
end

dlmwrite(fullfile("golden", "golden_fx.csv"), FX, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_fy.csv"), FY, "precision", "%.17g");
dlmwrite(fullfile("golden", "golden_vp.csv"), VP, "precision", "%.17g");
printf("goldens ecrits pour %d etats (fx, fy, vp)\n", N);
