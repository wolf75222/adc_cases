% golden_relax_gen.m -- paires in/out de relaxation15.m EXECUTEE (Octave) sur RIEMOM2D.
%
%   octave --no-gui --path /chemin/vers/RIEMOM2D golden_relax_gen.m
%
% Produit golden/golden_relax_in.csv, golden_relax_out.csv (N x 15, ordre
% [M00,M10,M20,M30,M40,M01,M11,M21,M31,M02,M12,M22,M03,M13,M04]) et
% golden_relax_meta.csv (N x 4 : lamin, Ma, branche, lambda_min(p2p2)).
% La colonne "branche" est un code recalcule ICI avec les fonctions DU DEPOT
% (M2CS4_15, p2p2_2D) -- elle sert a asserter la COUVERTURE cote Python, la
% verite in/out venant de relaxation15 elle-meme :
%   0 = identite (lambda_min > lamin, aucune projection)
%   1 = clamp |s30| ou |s03| > 4 + Ma/2
%   2 = bord univarie (H20 ou H02 < 1e-6)
%   3 = clamp s11 (|s11| >= 1 - 1e-6, substitution complete)
%   4 = projection collision15 (lambda_min <= lamin, cas general)
% Les vecteurs sont choisis HORS du fil du rasoir des seuils (pas de |s11| a
% 1e-7 pres de la coupure, etc.) pour que le port Python reproduise les MEMES
% branches malgre des arrondis differents.

lamin = 1.d-12;

% --- etats = moments de melanges discrets f = sum w_k delta(v - v_k), realisables
% par construction (memes generateurs que gen_states.py, valeurs distinctes) ---
function M = mix_moments(W, VX, VY)
  M = zeros(15,1);
  pq = [0 0;1 0;2 0;3 0;4 0;0 1;1 1;2 1;3 1;0 2;1 2;2 2;0 3;1 3;0 4];
  for k = 1:15
    p = pq(k,1); q = pq(k,2);
    M(k) = sum(W(:) .* (VX(:).^p) .* (VY(:).^q));
  end
end

states = {};   % {M, Ma}

% (0) etats sains STRICTEMENT interieurs : gaussienne discretisee sur une grille riche
% (un melange de 3-4 points est MARGINALEMENT realisable : lambda_min(p2p2) ~ 0 <= lamin
% declenche la projection ; il faut une distribution etalee pour lambda_min >> 1e-12)
function M = gauss_grid(u0, v0, sx, sy)
  [VX, VY] = meshgrid(linspace(-3,3,9)*sx + u0, linspace(-3,3,9)*sy + v0);
  W = exp(-((VX-u0).^2/(2*sx^2) + (VY-v0).^2/(2*sy^2)));
  W = W / sum(W(:));
  M = mix_moments(W, VX, VY);
end
states{end+1} = {gauss_grid(0.3, -0.2, 1.0, 0.7), 2};
states{end+1} = {gauss_grid(-0.5, 0.4, 0.8, 1.2), 20};
states{end+1} = {mix_moments([.3 .3 .2 .2], [1.2 -.8 .3 -.5], [.4 -.6 1.1 -.9]), 2};

% (1) |s30| enorme : poids minuscule tres loin en vx (skew geant), Ma = 2 -> S3m = 5
states{end+1} = {mix_moments([.495 .495 .01], [-.1 .1 30], [.5 -.5 .1]), 2};
states{end+1} = {mix_moments([.495 .495 .01], [.5 -.5 .1], [-.1 .1 30]), 2};  % symetrique en y

% (2) bord univarie : DEUX points en vx (H20 = 0), vy etale
states{end+1} = {mix_moments([.3 .3 .2 .2], [1 -1 1 -1], [.8 -.4 -.9 .5]), 2};

% (3) s11 -> +-1 : points presque alignes sur v = u (et v = -u)
states{end+1} = {mix_moments([.4 .4 .2], [1 -1 .3], [1.000001 -.999999 .300002]), 2};
states{end+1} = {mix_moments([.4 .4 .2], [1 -1 .3], [-1.000001 .999999 -.300002]), 2};

% (4) cas generaux non realisables a la marge : melanges asymetriques moderement
% pathologiques (l'un au moins doit declencher la projection collision15)
states{end+1} = {mix_moments([.6 .25 .15], [.2 -1.5 2.2], [-.3 1.1 .9]), 2};
states{end+1} = {mix_moments([.5 .3 .2], [0 2.5 -1.8], [.7 -.6 1.4]), 20};
states{end+1} = {mix_moments([.45 .35 .2], [3 -2 .5], [-.2 .8 -2.5]), 20};
% melange a fort aplatissement croise (cible Z1/Z2 potentiellement non realisable -> CJ)
states{end+1} = {mix_moments([.49 .49 .01 .01], [1 -1 8 -8], [-1 1 8 -8]), 2};

N = numel(states);
INM = zeros(N, 15); OUTM = zeros(N, 15); META = zeros(N, 4);
for t = 1:N
  M = states{t}{1}; Ma = states{t}{2};
  % code de branche, recalcule avec les fonctions DU DEPOT (jamais re-transcrites ici)
  [C4, S4] = M2CS4_15(M);
  s30 = S4(4); s40 = S4(5); s11 = S4(7); s21 = S4(8); s31 = S4(9);
  s12 = S4(11); s22 = S4(12); s03 = S4(13); s13 = S4(14); s04 = S4(15);
  S3m = 4 + Ma/2; small = 1.d-6;
  br = 0;
  if abs(s30) > S3m || abs(s03) > S3m
    br = 1;
  elseif (s40 - s30^2 - 1) < small || (s04 - s03^2 - 1) < small
    br = 2;
  elseif abs(s11) >= 1 - small
    br = 3;
  else
    p2 = p2p2_2D(s03,s04,s11,s12,s13,s21,s22,s30,s31,s40);
    lmin = min(real(eig(p2)));
    if max([0 lmin]) <= lamin
      br = 4;
    end
  end
  p2 = p2p2_2D(s03,s04,s11,s12,s13,s21,s22,s30,s31,s40);
  lmin = min(real(eig(p2)));
  Mout = relaxation15(M, lamin, Ma);
  INM(t,:) = M(:)'; OUTM(t,:) = Mout(:)'; META(t,:) = [lamin, Ma, br, lmin];
end

mkdir('golden');
dlmwrite('golden/golden_relax_in.csv',  INM,  'precision', '%.17g');
dlmwrite('golden/golden_relax_out.csv', OUTM, 'precision', '%.17g');
dlmwrite('golden/golden_relax_meta.csv', META, 'precision', '%.17g');
printf('golden_relax : %d etats, branches presentes = %s\n', N, mat2str(unique(META(:,3))'));
