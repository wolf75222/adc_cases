// Choix de la discretisation spatiale (flux) et de l'integrateur en temps AU NIVEAU
// du coupleur, sans toucher au modele :
//   - diocotron (scalaire, flux Rusanov) : le point d'entree unifie step<Disc, Integ>
//     reproduit advance pour SSPRK2 et donne un resultat different (mais conservatif)
//     pour SSPRK3.
//   - Euler-Poisson (systeme) : on choisit le flux Riemann (Rusanov vs HLLC) au niveau
//     du coupleur via advance<Limiter, Policy, NumericalFlux> ; les deux conservent la
//     masse et donnent des resultats reellement differents.
// HLL/HLLC exigent un modele a structure d'ondes (pressure/wave_speeds) : ils ne
// s'appliquent donc pas au diocotron scalaire, et c'est correct.

#include <adc/coupling/coupler.hpp>
#include <adc/integrator/time_integrator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/spatial_discretisation.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static double max_diff(const MultiFab& A, const MultiFab& B, const Box2D& dom,
                       int comp) {
  const Fab2D& a = A.fab(0);
  const Fab2D& b = B.fab(0);
  double d = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      d = std::max(d, std::fabs(a(i, j, comp) - b(i, j, comp)));
  return d;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // ============================================================
  // 1. Diocotron (scalaire) : integrateur en temps selectionnable.
  // ============================================================
  {
    const int n = 64;
    Box2D dom = Box2D::from_extents(n, n);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba = BoxArray::from_domain(dom, n);
    DistributionMapping dm(ba.size(), n_ranks());
    Diocotron model;
    model.B0 = 1.0;
    model.n_i0 = 1.0;
    model.alpha = 1.0;
    BCRec bc;
    const Real dx = geom.dx();
    const int nsteps = 20;

    auto init = [&](MultiFab& U) {
      Array4 a = U.fab(0).array();
      for_each_cell(dom, [a, geom](int i, int j) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        auto blob = [&](double cx, double cy) {
          const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          return std::exp(-r2 / 0.01);
        };
        a(i, j, 0) = 1.0 + 0.5 * blob(0.35, 0.5) + 0.5 * blob(0.65, 0.5);
      });
    };

    MultiFab Uref(ba, dm, 1, 2), Ustep2(ba, dm, 1, 2), Us3(ba, dm, 1, 2);  // VanLeer : 2 ghosts
    init(Uref);
    init(Ustep2);
    init(Us3);
    const Real m0 = sum(Uref);

    Coupler<Diocotron> c1(model, geom, ba, bc, bc), c2(model, geom, ba, bc, bc),
        c3(model, geom, ba, bc, bc);
    for (int s = 0; s < nsteps; ++s) c1.advance<VanLeer>(Uref, 0.5 * dx);
    for (int s = 0; s < nsteps; ++s)
      c2.step<MusclVanLeer, SSPRK2>(Ustep2, 0.5 * dx);
    for (int s = 0; s < nsteps; ++s)
      c3.step<MusclVanLeer, SSPRK3>(Us3, 0.5 * dx);

    // step<MusclVanLeer, SSPRK2> == advance<VanLeer> : le point d'entree unifie
    // ne change rien au schema, il l'enrobe.
    chk(max_diff(Uref, Ustep2, dom, 0) == 0.0, "step_ssprk2_matches_advance");
    chk(std::fabs(sum(Us3) - m0) < 1e-8, "ssprk3_mass");
    chk(max_diff(Uref, Us3, dom, 0) > 1e-9, "ssprk3_differs");
  }

  // ============================================================
  // 2. Euler-Poisson (systeme) : flux Riemann selectionnable.
  // ============================================================
  {
    const int N = 64;
    const double L = 1.0, rho0 = 1.0, p0 = 1.0, gamma = 5.0 / 3.0;
    const double four_pi_G = 20.0, eps = 1e-3, k = 2 * kPi / L;
    const double cs2 = gamma * p0 / rho0;
    Box2D dom = Box2D::from_extents(N, N);
    Geometry geom{dom, 0.0, L, 0.0, L};
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);
    EulerPoisson model;
    model.hydro.gamma = gamma;
    model.four_pi_G = four_pi_G;
    model.rho0 = rho0;
    BCRec bcU, bcPhi;
    const double dt = 0.4 * (L / N) / (std::sqrt(cs2) + 0.1);

    auto init = [&](MultiFab& U) {
      Fab2D& f = U.fab(0);
      const Box2D v = U.box(0);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const double drho = eps * rho0 * std::cos(k * geom.x_cell(i));
          f(i, j, 0) = rho0 + drho;
          f(i, j, 1) = 0.0;
          f(i, j, 2) = 0.0;
          f(i, j, 3) = (p0 + cs2 * drho) / (gamma - 1);
        }
    };

    MultiFab Ur(ba, dm, 4, 2), Uh(ba, dm, 4, 2);
    init(Ur);
    init(Uh);
    const double mass0 = sum(Ur, 0);

    Coupler<EulerPoisson> cr(model, geom, ba, bcU, bcPhi),
        ch(model, geom, ba, bcU, bcPhi);
    for (int s = 0; s < 30; ++s) {
      cr.advance<VanLeer, PerStageCoupling, RusanovFlux>(Ur, dt);
      ch.advance<VanLeer, PerStageCoupling, HLLCFlux>(Uh, dt);
    }

    chk(std::fabs(sum(Ur, 0) - mass0) < 1e-8, "ep_rusanov_mass");
    chk(std::fabs(sum(Uh, 0) - mass0) < 1e-8, "ep_hllc_mass");
    // Rusanov (diffusif) != HLLC (resout le contact) : le choix de flux est reel.
    chk(max_diff(Ur, Uh, dom, 0) > 1e-9, "flux_choice_differs");
  }

  if (fails == 0) std::printf("OK test_coupler_scheme\n");
  return fails == 0 ? 0 : 1;
}
