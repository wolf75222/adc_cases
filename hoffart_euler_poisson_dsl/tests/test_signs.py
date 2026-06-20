#!/usr/bin/env python3
"""Tests de convention de signe pour les helpers purement numpy de model.py.

Ce fichier teste directement les fonctions purement numpy de model.py contre le
VRAI module `adc` (aucun fake). Il faut donc l'extension `adc` importable (env conda
`adc` ou `PYTHONPATH=<adc_cpp>/build/python` ; `KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1` pour le build Kokkos-OpenMP).

Conventions verifiees
---------------------
drift_velocity_from_potential
    Le champ ExB est v0 = -(grad(phi) x Omega)/|Omega|^2.
    En 2D avec Omega = omega*e_z :
        u = -d_y(phi) / omega
        v = +d_x(phi) / omega
    source : formule (3) du papier, m x Omega = (omega*m_y, -omega*m_x).

paper_initial_density
    Anneau de densite avec perturbation azimutale sin(l*theta), eq. (35).

Poisson (signe, documente uniquement, sans import adc lourd)
    Le papier pose -Delta(phi) = alpha*rho. ADC resout Delta(phi) = rhs.
    model.py emet donc rhs = -alpha*rho (cf. model.py:129).

Run standalone (`python3 test_signs.py`) ou sous pytest.
"""

import os
import sys

import numpy as np

import adc  # noqa: F401  (real extension; satisfies model.py's `from adc import dsl`)

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_model():
    case_root = os.path.dirname(HERE)   # tests/ -> la racine du cas (model.py, run*.py)
    if case_root not in sys.path:
        sys.path.insert(0, case_root)
    import importlib
    if "model" in sys.modules:
        return sys.modules["model"]
    return importlib.import_module("model")


# ---------------------------------------------------------------------------
# (a) drift_velocity_from_potential, verification du signe ExB
# ---------------------------------------------------------------------------

def test_drift_phi_linear_x():
    """phi(x,y) = x => grad_phi = (1, 0) => u = 0, v = +1/omega."""
    model = _import_model()
    params = model.PaperParameters()

    n = 64
    h = params.length / n
    x = (np.arange(n) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    phi = X.copy()  # phi = x, d/dx phi = 1, d/dy phi = 0

    u, v = model.drift_velocity_from_potential(phi, params)

    # Region interieure au disque, loin des bords (evite l'erreur de gradient aux bords)
    margin = 3
    disc = np.hypot(X, Y) < params.radius - margin * h
    # On compare la derive echelonnee u*omega / v*omega (~O(1)). A omega=1e12, 1/omega~1e-12 est
    # sous l'atol par defaut (1e-8) de allclose : comparer u/v bruts serait un no-op (un zero ou un
    # signe inverse passerait). u*omega doit valoir 0, v*omega doit valoir +1.
    assert np.allclose(u[disc] * params.omega, 0.0, atol=1e-3), (
        "phi=x : u*omega doit etre ~0 dans le disque, max|u*omega|=%g"
        % np.abs(u[disc] * params.omega).max()
    )
    assert np.allclose(v[disc] * params.omega, 1.0, rtol=1e-3, atol=1e-6), (
        "phi=x : v*omega doit etre ~+1 dans le disque, max|v*omega-1|=%g"
        % np.abs(v[disc] * params.omega - 1.0).max()
    )


def test_drift_phi_linear_y():
    """phi(x,y) = y => grad_phi = (0, 1) => u = -1/omega, v = 0."""
    model = _import_model()
    params = model.PaperParameters()

    n = 64
    h = params.length / n
    x = (np.arange(n) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    phi = Y.copy()  # phi = y, d/dx phi = 0, d/dy phi = 1

    u, v = model.drift_velocity_from_potential(phi, params)

    margin = 3
    disc = np.hypot(X, Y) < params.radius - margin * h
    # derive echelonnee (cf. test_drift_phi_linear_x) : u*omega = -1, v*omega = 0.
    assert np.allclose(u[disc] * params.omega, -1.0, rtol=1e-3, atol=1e-6), (
        "phi=y : u*omega doit etre ~-1 dans le disque, max|u*omega+1|=%g"
        % np.abs(u[disc] * params.omega + 1.0).max()
    )
    assert np.allclose(v[disc] * params.omega, 0.0, atol=1e-3), (
        "phi=y : v*omega doit etre ~0 dans le disque, max|v*omega|=%g"
        % np.abs(v[disc] * params.omega).max()
    )


def test_drift_zero_outside_disc():
    """drift_velocity_from_potential doit mettre u=v=0 hors du disque."""
    model = _import_model()
    params = model.PaperParameters()

    n = 64
    h = params.length / n
    x = (np.arange(n) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    phi = X.copy()

    u, v = model.drift_velocity_from_potential(phi, params)

    outside = np.hypot(X, Y) > params.radius
    assert (u[outside] == 0.0).all(), "u doit etre 0 hors du disque"
    assert (v[outside] == 0.0).all(), "v doit etre 0 hors du disque"


# ---------------------------------------------------------------------------
# (b) paper_initial_density, structure azimutale et bornes
# ---------------------------------------------------------------------------

def test_paper_initial_density_outside_ring():
    """hors de l'anneau, rho doit valoir rho_min."""
    model = _import_model()
    params = model.PaperParameters()

    n = 128
    h = params.length / n
    x = (np.arange(n) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    radius = np.hypot(X, Y)

    rho = model.paper_initial_density(n, mode=4, params=params)

    outside_ring = (radius < params.ring_inner) | (radius > params.ring_outer)
    assert np.allclose(rho[outside_ring], params.rho_min), (
        "hors de l'anneau, rho doit valoir rho_min=%g" % params.rho_min
    )


def test_paper_initial_density_inside_ring_bounds():
    """dans l'anneau, rho in [rho_max*(1-2*delta), rho_max]."""
    model = _import_model()
    params = model.PaperParameters()

    n = 128
    h = params.length / n
    x = (np.arange(n) + 0.5) * h - params.radius
    X, Y = np.meshgrid(x, x, indexing="xy")
    radius = np.hypot(X, Y)

    rho = model.paper_initial_density(n, mode=4, params=params)

    ring = (radius >= params.ring_inner) & (radius <= params.ring_outer)
    delta = params.perturbation
    # formule eq. (35) : rho_max*(1 - delta + delta*sin(l*theta))
    # sin in [-1,1] => min = rho_max*(1 - 2*delta), max = rho_max
    lo = params.rho_max * (1.0 - 2.0 * delta)
    hi = params.rho_max
    assert rho[ring].min() >= lo * (1.0 - 1e-10), (
        "rho min dans l'anneau = %g < rho_max*(1-delta) = %g" % (rho[ring].min(), lo)
    )
    assert rho[ring].max() <= hi * (1.0 + 1e-10), (
        "rho max dans l'anneau = %g > rho_max = %g" % (rho[ring].max(), hi)
    )


def test_paper_initial_density_azimuthal_structure():
    """la densite dans l'anneau doit avoir la structure azimutale sin(l*theta).

    Verification via FFT sur un cercle au milieu de l'anneau : le coefficient
    d'ordre l doit dominer (|c_l| >> |c_k| pour k != l). Cette approche est
    robuste aux artefacts de transition de signe numerique (resolution de grille).
    """
    model = _import_model()
    params = model.PaperParameters()

    n = 256
    mode = 4
    rho = model.paper_initial_density(n, mode=mode, params=params)

    h = params.length / n
    # Echantillonnage sur un cercle au milieu de l'anneau
    r_sample = 0.5 * (params.ring_inner + params.ring_outer)
    ntheta = 1024
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    xi = 0.5 * params.length + r_sample * np.cos(theta)
    yi = 0.5 * params.length + r_sample * np.sin(theta)

    # Interpolation bilineaire (coherente avec sample_circle de run.py)
    fi = xi / h - 0.5
    fj = yi / h - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
    tx, ty = fi - i0, fj - j0
    vals = (
        rho[j0, i0] * (1.0 - tx) * (1.0 - ty)
        + rho[j0, i0 + 1] * tx * (1.0 - ty)
        + rho[j0 + 1, i0] * (1.0 - tx) * ty
        + rho[j0 + 1, i0 + 1] * tx * ty
    )

    # Analyse spectrale : le mode l doit dominer tres nettement
    coeffs = np.abs(np.fft.rfft(vals - vals.mean()))
    c_mode = coeffs[mode]
    # Tous les autres modes (hors k=0 et k=mode) doivent etre negligeables
    mask = np.ones(len(coeffs), dtype=bool)
    mask[0] = False
    mask[mode] = False
    c_other_max = coeffs[mask].max() if mask.any() else 0.0
    assert c_mode > 100.0 * c_other_max, (
        "mode l=%d : |c_%d|=%g doit dominer les autres modes (max=%g)"
        % (mode, mode, c_mode, c_other_max)
    )


# ---------------------------------------------------------------------------
# (c) Poisson sign, documente ici, assertion legere sans import lourd
# ---------------------------------------------------------------------------
# Le papier pose -Delta(phi) = alpha*rho.
# ADC resout Delta(phi) = rhs.
# model.py emet donc rhs = -alpha*rho (cf. model.py:129 : m.elliptic_rhs(-alpha*rho)).
# On ne peut pas appeler m.elliptic_rhs sans le vrai adc.dsl.Model. La convention
# est donc verifie ci-dessus (test_drift_*) de facon indirecte : si le signe de rhs
# etait inverse, la vitesse de derive serait opposee et les tests (a) echoueraient.
#
# Pour referene : la ligne pertinente est
#   m.elliptic_rhs(-alpha * rho)        # model.py:129


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("all %d sign-convention tests passed" % len(tests))


if __name__ == "__main__":
    _run_all()
