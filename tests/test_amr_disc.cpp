// Discretisation spatiale selectionnable AUSSI en AMR : AmrCouplerMP::step<Disc>()
// route le limiteur + le flux jusqu'a advance_amr (avant, NoSlope/Rusanov etaient codes
// en dur). On verifie que step<MusclMinmod>() tourne, conserve la masse AMR, et donne un
// resultat DIFFERENT de step() (ordre 1 par defaut) : le choix de schema agit vraiment.

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/spatial_discretisation.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool ok, const char* w) {
    if (!ok) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  BCRec bc;
  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = 1.0;

  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
  };
  const int CI0 = nc / 4, CI1 = 3 * nc / 4 - 1, CJ0 = nc / 4, CJ1 = 3 * nc / 4 - 1;
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  auto build = [&]() {
    MultiFab Uc(ba, dm, 1, 2), Uf(BoxArray(std::vector<Box2D>{fbox}), dm, 1, 2);  // 2 ghosts (MUSCL)
    auto fill = [&](MultiFab& U, double dx, double dy) {
      for (int li = 0; li < U.local_size(); ++li) {
        Array4 u = U.fab(li).array();
        const Box2D g = U.fab(li).grown_box();
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            u(i, j, 0) = ne0((i + 0.5) * dx, (j + 0.5) * dy);
      }
    };
    fill(Uc, dxc, dyc);
    fill(Uf, dxf, dyf);
    mf_average_down_mb(Uf, Uc);
    std::vector<AmrLevelMP> L;
    L.push_back({std::move(Uc), nullptr, dxc, dyc});
    L.push_back({std::move(Uf), nullptr, dxf, dyf});
    return AmrCouplerMP<Diocotron>(model, geom, ba, bc, std::move(L));
  };

  AmrCouplerMP<Diocotron> a = build(), b = build();
  a.update();
  b.update();
  const double m0 = a.mass();
  const double dt = 0.4 * dxc / a.max_drift_speed();
  for (int s = 0; s < 20; ++s) {
    a.step(dt);                  // FirstOrder par defaut (NoSlope + Rusanov)
    b.step<MusclMinmod>(dt);     // MUSCL Minmod + Rusanov
  }

  chk(std::fabs(a.mass() - m0) < 1e-10, "default_mass_conserved");
  chk(std::fabs(b.mass() - m0) < 1e-10, "muscl_mass_conserved");

  double maxdiff = 0;
  bool finite = true;
  const ConstArray4 ua = a.coarse().fab(0).const_array();
  const ConstArray4 ub = b.coarse().fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) {
      maxdiff = std::fmax(maxdiff, std::fabs(ua(i, j, 0) - ub(i, j, 0)));
      if (!std::isfinite(ub(i, j, 0))) finite = false;
    }
  std::printf("  AMR : max|NoSlope - MusclMinmod| = %.3e\n", maxdiff);
  chk(finite, "muscl_finite");
  chk(maxdiff > 1e-6, "amr_scheme_choice_acts");

  if (fails == 0) std::printf("OK test_amr_disc\n");
  return fails == 0 ? 0 : 1;
}
