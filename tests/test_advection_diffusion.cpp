// Modele applicatif AdvectionDiffusion : d_t u + a.grad u = nu Lap u. Valide que la
// diffusion (declaree par diffusivity()) s'ajoute au transport sans toucher au coeur :
//   - nu = 0 : advection pure, le pic d'une gaussienne se conserve (a la diffusion
//     numerique du schema pres).
//   - nu > 0 : le pic decroit (etalement diffusif), masse conservee dans les deux cas.

#include <adc/core/physical_model.hpp>
#include <adc/integrator/ssprk.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/advection_diffusion.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <array>
#include <cmath>
#include <cstdio>

using namespace adc;

static_assert(PhysicalModel<AdvectionDiffusion>, "AdvectionDiffusion : PhysicalModel");
static_assert(DiffusiveModel<AdvectionDiffusion>, "AdvectionDiffusion : diffusif");

static double peak(const MultiFab& U, const Box2D& dom) {
  const Fab2D& f = U.fab(0);
  double m = -1e300;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) m = std::max(m, f(i, j, 0));
  return m;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 96;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  const double dx = geom.dx();
  const double nu = 0.005;
  // dt respecte les DEUX contraintes : CFL advection (~0.25 dx) ET stabilite
  // parabolique dt < dx^2/(4 nu) (la diffusion restreint le pas, comme l'a note le
  // tuteur). On prend le min, partage par les deux runs pour un meme temps final.
  const double dt = std::min(0.25 * dx, 0.2 * dx * dx / nu);
  const int K = 150;

  auto run = [&](double nu) {
    AdvectionDiffusion model;
    model.ax = 1.0;
    model.ay = 0.0;
    model.nu = nu;
    MultiFab U(ba, dm, 1, 2), aux(ba, dm, 3, 1);
    aux.set_val(0.0);
    Array4 a = U.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      const double x = geom.x_cell(i), y = geom.y_cell(j);
      const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
      a(i, j, 0) = 1.0 + std::exp(-r2 / 0.005);
    });
    const double m0 = sum(U), p0 = peak(U, dom);
    for (int s = 0; s < K; ++s) advance_ssprk2<VanLeer>(model, U, aux, geom, bc, dt);
    return std::array<double, 3>{sum(U) - m0, peak(U, dom), p0};
  };

  const auto a0 = run(0.0);     // advection pure
  const auto ad = run(nu);      // advection + diffusion

  chk(std::fabs(a0[0]) < 1e-8, "advect_mass_conserved");
  chk(std::fabs(ad[0]) < 1e-8, "advdiff_mass_conserved");
  // le terme diffusif abaisse le pic davantage que l'advection seule.
  std::printf("  peak: advection=%.4f  adv+diff=%.4f  (initial=%.4f)\n", a0[1], ad[1],
              a0[2]);
  chk(ad[1] < a0[1] - 1e-3, "diffusion_lowers_peak");
  chk(ad[1] > 1.0, "stays_positive_above_background");

  if (fails == 0) std::printf("OK test_advection_diffusion\n");
  return fails == 0 ? 0 : 1;
}
