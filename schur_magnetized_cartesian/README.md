# Cas `schur_magnetized_cartesian` : effet temporel du complement de Schur

Mesure de l'effet TEMPOREL de l'etage source condense par Schur
(`adc.CondensedSchur`, pile #118-128) sur un fluide isotherme magnetise CARTESIEN
RAIDE. Script : [`run.py`](run.py).

C'est la mesure PR6 : le chemin polaire (diocotron) est explicite-only, donc cette
mesure est CARTESIENNE. On compare, sur le MEME modele :

- `explicit` : la force de Lorentz `m x Omega` est emise dans le C++ genere et
  avancee EXPLICITEMENT apres le transport ;
- `schur`    : la source locale est nulle ; l'etage electrostatique/Lorentz est
  avance IMPLICITEMENT par `CondensedSchur` (`set_source_stage`).


## Le terme raide et ce que le Schur change

La rotation cyclotronique `m x Omega` (Omega = omega_c e_z, omega_c = `B_z`)
integree EXPLICITEMENT impose la borne `dt * omega_c < O(1)` : a omega_c grand le
pas explicite s'effondre. `CondensedSchur` assemble et resout l'operateur condense
`A = I + theta^2 dt^2 alpha rho B^{-1}` (B portant la rotation de Lorentz) et
avance la source implicitement : la borne cyclotronique disparait, le pas n'est
plus limite que par le transport hyperbolique.

Pour isoler la raideur de la SOURCE on prend une vitesse du son lente
(`cs2 = 1e-4`) : le pas de transport explicite ~ h / cs reste large devant
1 / omega_c. La methode mesure, pour chaque variante, le plus grand `dt` stable
(densite finie, bornee, positive jusqu'a `t_end`) par balayage geometrique.


## Resultats

omega_c (B_z) = 1000, cs2 = 1e-4, alpha = 1, n = 16, L = 1 (h = 0.0625),
t_end = 1.0, transport minmod / Rusanov, etage source = Crank-Nicolson
(theta = 0.5) et Euler retrograde (theta = 1.0).

Pas de transport limitant ~ 6.2e-2 ; borne explicite de la source ~ 1/omega_c =
1.0e-3.

    methode                                  dt_stable     dt*omega_c
    explicit (Lorentz explicite)              3.162e-04           0.32
    schur theta=0.5 (Crank-Nicolson)          1.778e-01         177.83
    schur theta=1.0 (Euler retrograde)        3.162e-01         316.23

    gain en pas de temps du Schur sur l'explicite :
      theta=0.5 -> 562x ; theta=1.0 -> 1000x

Lecture :

- L'explicite plafonne a `dt * omega_c ~ 0.3` : c'est la borne de stabilite de la
  rotation cyclotronique explicite, conforme a dt * omega_c < O(1).
- Le Schur reste stable a `dt * omega_c` de 178 (theta=0.5) a 316 (theta=1.0),
  bien au-dela de la borne explicite : il a retire la contrainte cyclotronique.
  Le pas Schur (~ 0.18 a 0.32) approche le pas de transport (~ 0.06), preuve que
  c'est maintenant le transport, plus la source, qui limite.
- theta=1.0 (Euler retrograde, inconditionnellement stable pour la rotation) gagne
  davantage que theta=0.5 (Crank-Nicolson, marginalement stable), comme attendu.


## Note de plateforme

Le backend DSL `production` (natif zero-copie) echoue au dlopen sur macOS arm64
avec ce build ; le cas utilise donc le backend `aot` (host-marshale), qui supporte
`set_source_stage` et la force de Lorentz via `B_z`. Le chemin AOT n'expose que
l'integrateur explicite SSPRK2 pour le transport (pas SSPRK3) : sans incidence sur
la conclusion temporelle, qui porte sur l'etage SOURCE.

`adc.Split(Explicit, CondensedSchur)` n'est cable que par le chemin natif
production (l'ABI AOT ne transporte pas SSPRK3) ; le cas branche donc l'etage
condense directement via `set_source_stage`, qui execute le MEME C++
(`CondensedSchurSourceStepper`, #126).


## Lancer

    PYTHONPATH=<adc_cpp>/build-master/python \
        python schur_magnetized_cartesian/run.py --csv

Options : `--n`, `--omega-c`, `--cs2`, `--alpha`, `--t-end`, `--csv`
(ecrit `out/schur_magnetized_cartesian/dt_stable.csv`).
