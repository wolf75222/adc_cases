#!/usr/bin/env python3
"""Figures de TIMING pour schur_magnetized_cartesian (categorie experimental).

Pas de figure physique : un cas de TIMING ne montre pas un champ mais le PLUS GRAND
dt stable par methode (explicite vs Schur condense). Deux panneaux :

  (1) timing_dt_stable.png : barres log du dt_stable par methode au point de
      reference documente (omega_c=1e3, t_end=1.0) lu dans dt_stable.csv ; annote
      le produit dt*omega_c et le gain. La barre explicite plafonne a la borne
      cyclotronique dt*omega_c ~ O(0.3) ; les barres Schur la franchissent.

  (2) timing_vs_omega.png : dt_stable et dt*omega_c vs omega_c pour explicite et
      Schur, depuis une mesure FRAICHE ciblee (3 valeurs de omega_c, t_end court).
      La courbe explicite suit dt ~ 1/omega_c (produit borne) ; la courbe Schur est
      PLATE (independante de omega_c), calee sur le pas de transport.

Les nombres sont LUS, pas inventes :
  - panneau (1) <- dt_stable.csv (run complet documente, omega_c=1e3, t_end=1.0) ;
  - panneau (2) <- /tmp/schur_measure.json (mesure ciblee de ce passage).
Si une source manque, le panneau correspondant est saute (pas de valeur fabriquee).

Sortie : figures/*.png + figures/provenance.json (memes champs que les autres cas).
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Source 1 : CSV documente (run complet : omega_c=1e3, t_end=1.0).
CSV_CANDIDATES = [
    os.path.join(HERE, "out", "dt_stable.csv"),
    "/Users/romaindespoulain/Documents/Stage_Romain/adc_cases/out/"
    "schur_magnetized_cartesian/dt_stable.csv",
]
# Source 2 : mesure fraiche ciblee (3 omega_c, t_end court).
MEASURE_JSON = "/tmp/schur_measure.json"

OMEGA_C_REF = 1.0e3  # omega_c du run complet documente (CSV)


def sha(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def load_csv():
    for p in CSV_CANDIDATES:
        if os.path.exists(p):
            rows = []
            with open(p) as f:
                for r in csv.DictReader(f):
                    rows.append((r["method"], float(r["dt_stable"]),
                                 float(r["dt_times_omega_c"]),
                                 r["gain_over_explicit"]))
            return p, rows
    return None, None


def fig_bars(csv_path, rows):
    """Panneau (1) : barres log dt_stable par methode (point de reference CSV)."""
    short = ["explicite\n(Lorentz explicite)",
             "Schur theta=0.5\n(Crank-Nicolson)",
             "Schur theta=1.0\n(Euler retrograde)"]
    dts = [r[1] for r in rows]
    prods = [r[2] for r in rows]
    colors = ["#c44e52", "#4c72b0", "#55a868"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(range(len(dts)), dts, color=colors, width=0.6, log=True)
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("plus grand dt stable (echelle log)")
    ax.set_title("dt stable par methode (omega_c = %g, t_end = 1.0, n=16, cs2=1e-4)"
                 % OMEGA_C_REF, fontsize=10)
    de = dts[0]
    for b, dt, prod in zip(bars, dts, prods):
        gain = dt / de if de > 0 else float("inf")
        ax.annotate("dt=%.3e\ndt*wc=%.1f\ngain=%.0fx"
                    % (dt, prod, gain),
                    (b.get_x() + b.get_width() / 2, dt),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8)
    # ligne de la borne cyclotronique explicite dt = O(0.3)/omega_c
    ax.axhline(0.3 / OMEGA_C_REF, ls="--", color="#555555", lw=1)
    ax.text(2.4, 0.3 / OMEGA_C_REF, " borne explicite\n dt*wc~0.3",
            va="center", ha="left", fontsize=8, color="#555555")
    ax.set_ylim(de / 3, max(dts) * 6)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "timing_dt_stable.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_vs_omega(meas):
    """Panneau (2) : dt_stable et dt*omega_c vs omega_c (mesure fraiche)."""
    res = meas["results"]
    wcs = sorted(float(k) for k in res)
    de = [res[str(w)]["explicit"] for w in wcs]
    ds = [res[str(w)]["schur10"] for w in wcs]
    pe = [res[str(w)]["prod_el"] for w in wcs]
    ps = [res[str(w)]["prod_10"] for w in wcs]
    tdt = meas["transport_dt"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # gauche : dt_stable vs omega_c (log-log). Explicite ~ 1/omega_c ; Schur plat.
    de_pl = [d if d > 0 else np.nan for d in de]
    a1.loglog(wcs, de_pl, "o-", color="#c44e52", label="explicite")
    a1.loglog(wcs, ds, "s-", color="#55a868", label="Schur theta=1.0")
    a1.axhline(tdt, ls="--", color="#555555", lw=1)
    a1.text(wcs[0], tdt, " pas de transport ~%.2e" % tdt,
            va="bottom", ha="left", fontsize=8, color="#555555")
    # pente -1 de reference (1/omega_c) calee sur le 1er point explicite fini
    w0 = next((w for w, d in zip(wcs, de) if d > 0), None)
    if w0 is not None:
        d0 = res[str(w0)]["explicit"]
        a1.loglog(wcs, [d0 * w0 / w for w in wcs], ":", color="#c44e52",
                  lw=1, label="pente -1 (1/omega_c)")
    # marque omega_c ou l'explicite est instable a tout dt (de=0)
    for w, d in zip(wcs, de):
        if d == 0:
            a1.annotate("explicite\ninstable\n(tout dt)", (w, ds[wcs.index(w)]),
                        textcoords="offset points", xytext=(-6, -38),
                        ha="center", fontsize=8, color="#c44e52")
    a1.set_xlabel("omega_c (= B_z)")
    a1.set_ylabel("plus grand dt stable")
    a1.set_title("dt stable : explicite ~ 1/omega_c, Schur plat", fontsize=10)
    a1.legend(fontsize=8)
    a1.grid(True, which="both", alpha=0.3)

    # droite : produit dt*omega_c vs omega_c. Explicite borne ; Schur explose.
    pe_pl = [p if p > 0 else np.nan for p in pe]
    a2.loglog(wcs, pe_pl, "o-", color="#c44e52", label="explicite")
    a2.loglog(wcs, ps, "s-", color="#55a868", label="Schur theta=1.0")
    a2.axhline(1.0, ls="--", color="#555555", lw=1)
    a2.text(wcs[0], 1.0, " dt*omega_c = 1 (borne O(1))",
            va="bottom", ha="left", fontsize=8, color="#555555")
    a2.set_xlabel("omega_c (= B_z)")
    a2.set_ylabel("dt_stable * omega_c")
    a2.set_title("produit dt*omega_c : explicite ~ O(1), Schur non borne",
                 fontsize=10)
    a2.legend(fontsize=8)
    a2.grid(True, which="both", alpha=0.3)

    fig.suptitle("Mesure fraiche ciblee (n=16, cs2=1e-4, t_end=%.2f, alpha=1)"
                 % meas["t_end"], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIGDIR, "timing_vs_omega.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    produced = []
    prov = {
        "script": "schur_magnetized_cartesian/make_figures.py",
        "command": "python make_figures.py",
        "produces": [],
        "adc_cpp_sha": sha("/Users/romaindespoulain/Documents/Stage_Romain/adc_cpp"),
        "adc_cases_sha": sha(HERE),
        "backend": "DSL 'aot' (host-marshale ; production non lie sur macOS arm64)",
        "resolution": "16x16",
        "cs2": 1.0e-4,
        "alpha": 1.0,
        "charge_q": -1.0,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "adc_module": __import__("adc").__file__,
        "platform": "macOS arm64 (Apple Silicon), serie mono-process",
    }

    csv_path, rows = load_csv()
    if rows is not None:
        produced.append(fig_bars(csv_path, rows))
        prov["panel1_source_csv"] = csv_path
        prov["panel1_omega_c"] = OMEGA_C_REF
        prov["panel1_t_end"] = 1.0
        prov["dt_stable_ref"] = {r[0]: r[1] for r in rows}
        prov["dt_times_omega_c_ref"] = {r[0]: r[2] for r in rows}
        prov["gain_ref"] = {r[0]: r[3] for r in rows}
    else:
        print("WARN : pas de dt_stable.csv ; panneau (1) saute")

    if os.path.exists(MEASURE_JSON):
        meas = json.load(open(MEASURE_JSON))
        produced.append(fig_vs_omega(meas))
        prov["panel2_source_json"] = MEASURE_JSON
        prov["panel2_t_end"] = meas["t_end"]
        prov["transport_dt"] = meas["transport_dt"]
        prov["measure_vs_omega"] = {
            k: {kk: v[kk] for kk in ("explicit", "schur05", "schur10",
                                     "prod_el", "prod_05", "prod_10",
                                     "gain05", "gain10")}
            for k, v in meas["results"].items()}
    else:
        print("WARN : pas de /tmp/schur_measure.json ; panneau (2) saute")

    prov["produces"] = [os.path.basename(p) for p in produced]
    with open(os.path.join(FIGDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    for p in produced:
        print("ecrit", p)
    print("ecrit", os.path.join(FIGDIR, "provenance.json"))


if __name__ == "__main__":
    main()
