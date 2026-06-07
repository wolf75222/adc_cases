#!/usr/bin/env python3
"""Cheap analytic oracle for the Hoffart magnetic Euler-Poisson DSL formulas."""

import numpy as np

from model import PaperParameters, magnetic_euler_poisson_model


def main():
    p = PaperParameters(beta=3.0, temperature=0.25)
    model = magnetic_euler_poisson_model(p, source="local")._m

    rho = np.array([[1.0, 2.0], [1.5, 0.8]])
    mx = np.array([[0.3, -0.2], [0.7, 0.1]])
    my = np.array([[-0.4, 0.6], [0.2, -0.5]])
    gx = np.array([[0.2, -0.1], [0.3, 0.4]])
    gy = np.array([[-0.3, 0.5], [0.1, -0.2]])
    U = np.stack([rho, mx, my])
    aux = {"phi": np.zeros_like(rho), "grad_x": gx, "grad_y": gy}

    fx = model.flux(U, aux, 0)
    fy = model.flux(U, aux, 1)
    src = model.source_value(U, aux)
    u, v = mx / rho, my / rho
    pressure = p.temperature * rho

    np.testing.assert_allclose(fx, np.stack([mx, mx * u + pressure, mx * v]))
    np.testing.assert_allclose(fy, np.stack([my, my * u, my * v + pressure]))
    np.testing.assert_allclose(
        src,
        np.stack([
            np.zeros_like(rho),
            -rho * gx + p.omega * my,
            -rho * gy - p.omega * mx,
        ]),
    )

    env = model._env(U, aux)
    np.testing.assert_allclose(model._elliptic.eval(env), -p.alpha * rho)
    assert model.max_wave_speed(U, aux, 0) > 0.0
    assert model.max_wave_speed(U, aux, 1) > 0.0
    print("OK Hoffart DSL: flux, Lorentz/electric source, eigenvalues and Poisson rhs")


if __name__ == "__main__":
    main()

