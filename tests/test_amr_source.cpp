// Le chemin AMR applique-t-il model.source ? Avant correctif, advance_amr ne faisait
// que -div(F) et IGNORAIT la source (invisible avec le diocotron dont source=0). Ici un
// modele a SOURCE CONSTANTE c et FLUX NUL : la solution exacte est u(t) = u0 + c t sur
// CHAQUE cellule de CHAQUE niveau (pas de transport, juste l'ODE locale). Le sous-cyclage
// doit donner le meme increment sur grossier (dt) et fin (2 x dt/2).

#include <adc/core/physical_model.hpp>
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// Source constante, aucun flux : du/dt = c.
struct ConstSource {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real c = 1.0;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{c}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};
static_assert(PhysicalModel<ConstSource>, "ConstSource : PhysicalModel");

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
  const double dxc = geom.dx(), dyc = geom.dy();
  DistributionMapping dm(1, 1), dm2(2, 1);
  BoxArray bac(std::vector<Box2D>{dom});
  // deux patchs fins couvrant l'interieur grossier [8,23] x [8,23] (coords fines).
  Box2D left{{16, 16}, {31, 47}}, right{{32, 16}, {47, 47}};
  BoxArray baf(std::vector<Box2D>{left, right});

  ConstSource model;
  model.c = 2.0;
  const double u0 = 1.0, dt = 0.01;
  const int K = 10;
  const double uT = u0 + model.c * (K * dt);  // solution exacte

  MultiFab axc(bac, dm, 3, 1), axf(baf, dm2, 3, 1);
  axc.set_val(0.0);
  axf.set_val(0.0);

  LevelHierarchy h;
  h.base_dom = dom;
  h.base_per = Periodicity{true, true};
  {
    MultiFab Uc(bac, dm, 1, 1), Uf(baf, dm2, 1, 1);
    Uc.set_val(u0);
    Uf.set_val(u0);
    h.levels.resize(2);
    h.levels[0] = {std::move(Uc), &axc, dxc, dyc};
    h.levels[1] = {std::move(Uf), &axf, dxc / 2, dyc / 2};
  }

  for (int s = 0; s < K; ++s) advance_amr<NoSlope, RusanovFlux>(model, h, dt);

  // niveau grossier ET niveau fin doivent valoir uT partout.
  double err = 0;
  for (int lvl = 0; lvl < 2; ++lvl) {
    const MultiFab& U = h.levels[lvl].U;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          err = std::fmax(err, std::fabs(u(i, j, 0) - uT));
    }
  }
  std::printf("  AMR source : u attendu=%.4f  err max=%.2e\n", uT, err);
  chk(err < 1e-12, "source_applied_all_levels");

  if (fails == 0) std::printf("OK test_amr_source\n");
  return fails == 0 ? 0 : 1;
}
