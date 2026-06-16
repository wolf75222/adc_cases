#!/usr/bin/env python3
"""Diagnostic de normalisation du taux de croissance diocotron (chemin polaire ExB).

But
---
Mesurer, sur le chemin polaire explicite d'ADC (anneau global, transport ExB
scalaire, WENO5 + SSPRK3, Poisson polaire Dirichlet), le taux de croissance
diocotron brut gamma_raw et la rotation brute Omega_raw du mode azimutal l, puis
montrer que la normalisation globale

    gamma_norm = gamma_raw * (2 pi / rhobar)

reproduit le taux du papier Hoffart et al. (arXiv:2510.11808), tandis qu'une
normalisation "par rotation locale" (g_rot ci-dessous) echoue.

Principe
--------
On suit le coefficient complexe c_l(t) du mode l de PHI sur le cercle interne
r = r0 :

    c_l(t) ~ exp(-i omega_l t)
    => |c_l| ~ exp(Im(omega) t),  arg(c_l) ~ -Re(omega) t.

Mesures de simulation :
    gamma_raw = pente de log|c_l|       (~ Im(omega) en unites ExB-naturelles)
    Omega_raw = -pente de arg(c_l)      (~ Re(omega))

Le rapport gamma_raw / Omega_raw = Im(omega)/Re(omega) est invariant d'echelle
(propriete intrinseque du mode).

Deux normalisations sont calculees et comparees a la cible papier :

1. g_2pi  = gamma_raw * 2 pi / rhobar
   la trouvaille. gamma_raw est deja Im(omega) en unites de temps ExB-naturelles
   (aucun re-scaling beta) ; le facteur 2 pi / rhobar est le facteur global qui
   amene les unites a celles du papier. Avec rhobar = rho_max = 1, c'est 2 pi.

2. g_rot  = (gamma_raw / |Omega_raw|) * |Re(omega)|_ana * 2 pi
   Normalisation "par rotation locale". Elle echoue ici parce qu'au bord interne
   r0 la rotation mesuree Omega_raw est ~ 0 (pas de charge enfermee a l'interieur
   de l'anneau, donc pas de rotation de corps rigide a r0). Le ratio explose et
   g_rot devient absurde : la normalisation correcte est le facteur global
   2 pi / rhobar, pas une rotation locale.

Cibles analytiques (eigenmode complexe, echelle papier 6:8:16, top-hat) :
    gamma_papier (Im normalise) : l=3 -> 0.772, l=4 -> 0.911, l=5 -> 0.683
    |Re(omega)|_ana             : l=3 -> 0.331, l=4 -> 0.439, l=5 -> 0.547
    ratio Im/Re analytique      : l=3 -> 0.371, l=4 -> 0.331, l=5 -> 0.200

Conclusion attendue (cf. ../docs/NORMALIZATION.md)
---------------------------------------------
g_2pi reproduit le papier (l=4 exact a n=128 et n=192 ; l=3 +26% ; l=5 oscille
selon la fenetre de fit, c'est de la sensibilite a la fenetre, pas un deficit
de physique). g_rot diverge (Omega_raw ~ 0 a r0).

Lancer
------
    PYTHONPATH=<adc_cpp>/build-master/python \
        python hoffart_euler_poisson_dsl/diag/diag_polar_omega.py [n]

n par defaut = 128. Le module `adc` doit etre construit (chemin polaire =
adc.System(mesh=adc.PolarMesh(...)) + transport ExB + Poisson polaire).
"""

from __future__ import annotations

import math
import sys

import numpy as np

import adc


# Geometrie de l'anneau (echelle papier 6:8:16). RMIN = bord interne du domaine
# polaire (l'anneau de densite vit dans [R0, R1], le domaine va de RMIN a RW).
R0, R1, RW, RMIN = 6.0, 8.0, 16.0, 2.0
RHO_MIN, RHO_MAX, DELTA = (
    1e-6,
    1.0,
    0.1,
)  # rhobar = RHO_MAX dans la normalisation 2pi/rhobar

PAPER = {3: 0.772, 4: 0.911, 5: 0.683}  # gamma normalise du papier
RE_ANA = {
    3: 0.33144,
    4: 0.43859,
    5: 0.54747,
}  # |Re(omega)| analytique (eigenmode complexe)
RATIO_ANA = {3: 0.3708, 4: 0.3309, 5: 0.1998}  # Im/Re analytique


def run(
    l: int, n: int, cfl: float = 0.4, nsteps: int = 2200
) -> tuple[np.ndarray, np.ndarray]:
    """Avance le mode l sur le chemin polaire et renvoie (temps, c_l(t)) a r=r0."""
    sim = adc.System(mesh=adc.PolarMesh(r_min=RMIN, r_max=RW, nr=n, ntheta=n))
    sim.add_block(
        "ne",
        model=adc.Model(
            state=adc.Scalar(),
            transport=adc.ExB(B0=1.0),
            source=adc.NoSource(),
            elliptic=adc.ChargeDensity(charge=1.0),
        ),
        spatial=adc.Spatial(weno5=True),
        time=adc.Explicit(method="ssprk3"),
    )
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")

    dr = (RW - RMIN) / n
    dth = 2 * math.pi / n
    rho = []
    for j in range(n):
        th = (j + 0.5) * dth
        dper = 1.0 - DELTA + DELTA * math.sin(l * th)
        for i in range(n):
            r = RMIN + (i + 0.5) * dr
            rho.append(RHO_MAX * dper if (R0 <= r <= R1) else RHO_MIN)
    sim.set_density("ne", rho)

    i_r0 = max(0, min(n - 1, int(round((R0 - RMIN) / dr - 0.5))))
    ts, cs = [], []
    for step in range(nsteps + 1):
        phi = (
            np.asarray(sim.potential(), float).ravel().reshape(n, n)
        )  # [theta, r]
        if not np.isfinite(phi).all():
            break
        ck = (np.fft.rfft(phi[:, i_r0]) / n)[l]
        ts.append(sim.time())
        cs.append(ck)
        if step == nsteps:
            break
        sim.step_cfl(cfl)
    return np.array(ts), np.array(cs)


def measure(l: int, n: int) -> dict | None:
    """Mesure gamma_raw, Omega_raw, ratio, et les deux normalisations (g_2pi, g_rot)."""
    ts, cs = run(l, n)
    mag = np.abs(cs)
    good = mag > 0
    ts, cs, mag = ts[good], cs[good], mag[good]
    if len(ts) < 10:
        return None

    # Fenetre de croissance exponentielle : de 1.3*a0 jusqu'a 0.6*pic (version
    # validee n=128). La sensibilite a cette fenetre explique le scatter l=5.
    a0 = mag[0]
    pk = int(mag.argmax())
    lo = int(np.searchsorted(mag, 1.3 * a0))
    if pk > lo + 3:
        hi = int(np.searchsorted(mag[: pk + 1], 0.6 * mag[pk]))
    else:
        hi = len(ts) - 1
    hi = min(max(hi, lo + 4), len(ts) - 1)
    if hi - lo < 4:
        lo, hi = 1, len(ts) - 1
    sl = slice(lo, hi + 1)

    g_raw = np.polyfit(ts[sl], np.log(mag[sl]), 1)[0]
    ph = np.unwrap(np.angle(cs[sl]))
    om_raw = -np.polyfit(ts[sl], ph, 1)[0]
    ratio = g_raw / abs(om_raw) if om_raw else float("nan")

    # Normalisation globale 2pi/rhobar (la trouvaille) :
    g_2pi = g_raw * 2.0 * math.pi / RHO_MAX
    # Normalisation "par rotation locale" (echoue : Omega_raw ~ 0 a r0) :
    g_rot = (
        (g_raw / abs(om_raw)) * RE_ANA[l] * 2.0 * math.pi
        if om_raw
        else float("nan")
    )

    return dict(
        g_raw=g_raw,
        om_raw=om_raw,
        ratio=ratio,
        g_2pi=g_2pi,
        g_rot=g_rot,
        win=(float(ts[lo]), float(ts[hi])),
        tf=float(ts[-1]),
    )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    print(
        "DIAG rotation polaire n=%d (top-hat [%g,%g], R=%g, WENO5/SSPRK3)"
        % (n, R0, R1, RW)
    )
    print(
        "  normalisation = gamma_raw * 2pi/rhobar (rhobar=%g) ; cible = g_pap"
        % RHO_MAX
    )
    print(
        "%3s %9s %9s | %8s %8s | %9s %9s %9s"
        % (
            "l",
            "g_raw",
            "Om_raw",
            "ratio",
            "rat_ana",
            "g_2pi",
            "g_rot",
            "g_pap",
        )
    )
    for l in (3, 4, 5):
        m = measure(l, n)
        if m is None:
            print("%3d  echec (instable / non-fini)" % l)
            continue
        print(
            "%3d %9.5f %9.5f | %8.4f %8.4f | %9.4f %9.4f %9.4f  [win %.2f,%.2f tf=%.2f]"
            % (
                l,
                m["g_raw"],
                m["om_raw"],
                m["ratio"],
                RATIO_ANA[l],
                m["g_2pi"],
                m["g_rot"],
                PAPER[l],
                m["win"][0],
                m["win"][1],
                m["tf"],
            )
        )


if __name__ == "__main__":
    main()
