# Cas `magnetic_isothermal_dsl` : fluide isotherme magnetise ecrit en formules

Troisieme demonstrateur du plan declaratif, apres
[`diocotron_dsl`](../diocotron_dsl/run.py) (mono-espece) et
[`two_species_dsl`](../two_species_dsl/run.py) (multi-espece). Script : [`run.py`](run.py).

Tout le modele est ecrit en expressions symboliques `adc.dsl.Model` (variables conservatives,
primitives, flux, valeurs propres, source, second membre elliptique). `adc.dsl` genere le C++, le
compile et l'installe comme bloc via `add_equation(...)`. Aucune brique nommee, aucun modele natif
de reference n'existe pour ce modele.

## La nouveaute : une source qui lit le champ `B_z`

Les deux premiers demonstrateurs lisaient le contrat aux de base (`phi`, `grad phi`). Ici la source
lit en plus un champ auxiliaire ETENDU, `B_z`, et porte la force de Lorentz :

```python
rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v",
                                  roles=["Density", "MomentumX", "MomentumY"])
u = m.primitive("u", mx / rho); v = m.primitive("v", my / rho)
gx = m.aux("grad_x"); gy = m.aux("grad_y"); bz = m.aux("B_z")
cs2 = m.param("cs2", 1.0); q = m.param("charge", -1.0)
m.flux(x=[mx, mx*u + cs2*rho, mx*v], y=[my, my*u, my*v + cs2*rho])
m.eigenvalues(x=[u - dsl.sqrt(cs2), u, u + dsl.sqrt(cs2)],
              y=[v - dsl.sqrt(cs2), v, v + dsl.sqrt(cs2)])
m.source([0.0, q*rho*(-gx) + bz*my, q*rho*(-gy) - bz*mx])  # electrostatique + Lorentz v x B_z
m.elliptic_rhs(q * rho)
```

Le terme `B_z my` / `-B_z mx` est la projection 2D de `(q rho / c) v x B` avec `B = B_z e_z`
(constantes absorbees dans `B_z`) : il fait TOURNER la quantite de mouvement sans toucher a la masse.

## `B_z` pilote 100 % depuis Python

`B_z` est une composante canonique du canal `adc::Aux` (indice 3, au dela de `phi` / `grad phi`).
Un modele DSL qui lit `aux("B_z")` declare `n_aux = 4` ; `add_equation` elargit le canal aux
partage ; on peuple `B_z` par un appel deja existant du binding :

```python
sim.set_magnetic_field(B0 * np.ones((n, n)))   # champ constant ici
```

Aucune modification du coeur `adc_cpp` n'est requise.

## Validation (sans modele natif de reference)

1. **Parite inter-backend** : si les backends `production` (natif zero-copie) ET `aot` se lient sur
   la plateforme, leurs `eval_rhs` et leurs etats apres quelques pas sont BIT-IDENTIQUES
   (`np.array_equal`, `dmax == 0`), comme `diocotron_dsl` prouve l'equivalence DSL contre natif.
2. **Oracle Lorentz** : la difference de residu entre `B_z = B0` et `B_z = 0` vaut, sur la quantite
   de mouvement, EXACTEMENT `(B0 my, -B0 mx)` calcule en numpy (`dmax == 0`) ; a `B_z = 0` le terme
   s'annule. Cela prouve la lecture exacte de `B_z` et la bonne forme du terme.
3. **Evolution** : run court stable, fini, densite positive, masse conservee (derive < 1e-9).
4. **Rotation** : avec `B_z != 0`, la quantite de mouvement transverse `my`, initialement nulle,
   devient non nulle : la force de Lorentz devie bien l'ecoulement.

## Lancer

```bash
PYTHONPATH=<adc_cpp>/build-py/python python magnetic_isothermal_dsl/run.py
```

Le cas a besoin d'un compilateur C++20 (`needs = ["cxx"]`) pour generer la `.so` du modele.

Sur une plateforme ou le chemin natif ne peut pas etre charge (macOS, espace de noms a deux
niveaux), seul `aot` se lie : la parite inter-backend est alors sautee, mais la correction reste
prouvee par l'oracle analytique de Lorentz. En CI (Ubuntu), les deux backends se lient.
