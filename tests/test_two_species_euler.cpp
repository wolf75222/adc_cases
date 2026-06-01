// Cas canonique deux especes (TODO 2.4) : electrons Euler (4 var) + ions Euler
// ISOTHERMES (3 var) + Poisson de systeme, composes par CoupledSystem + SystemCoupler.
//
// Demontre que des blocs HETEROGENES (tailles d'etat differentes) partagent un meme
// champ phi dont le second membre est la densite de charge totale q_e n_e + q_i n_i
// (ChargeDensityRhs). On verifie : conservation de la masse par espece (continuite,
// domaine periodique), second membre de charge correct, et stabilite sur quelques pas.

#include <adc/core/coupled_system.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/model/charged_fluid.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

using ElectronBlock =
    EquationBlock<ChargedEuler, MusclMinmod, ExplicitTime<SSPRK2, 1>>;
using IonBlock =
    EquationBlock<ChargedEulerIsothermal, MusclMinmod, ExplicitTime<SSPRK2, 1>>;

static_assert(EquationBlockLike<ElectronBlock>);
static_assert(EquationBlockLike<IonBlock>);
static_assert(ElectronBlock::Model::n_vars == 4);
static_assert(IonBlock::Model::n_vars == 3);

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int n = 32;
  const Real L = Real(1);
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, double(L), 0.0, double(L)};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dx = geom.dx();
  BCRec bc;  // periodique partout

  MultiFab Ue(ba, dm, 4, 2), Ui(ba, dm, 3, 2);

  // IC quasi-neutre : n_i = 1 (uniforme), n_e = 1 + eps cos(k x) -> charge -eps cos(k x),
  // a moyenne nulle (Poisson periodique solvable). Vitesses nulles ; p_e = n_e (KE = 0).
  const Real eps = Real(0.01), gamma = Real(1.4), k = Real(2) * Real(M_PI) / L;
  {
    Array4 ue = Ue.fab(0).array();
    Array4 ui = Ui.fab(0).array();
    const Box2D g = Ue.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const Real x = (Real(i) + Real(0.5)) * dx;
        const Real ne = Real(1) + eps * std::cos(k * x);
        ue(i, j, 0) = ne;
        ue(i, j, 1) = 0;
        ue(i, j, 2) = 0;
        ue(i, j, 3) = ne / (gamma - Real(1));  // E = p/(g-1), p = ne, KE = 0
        ui(i, j, 0) = Real(1);
        ui(i, j, 1) = 0;
        ui(i, j, 2) = 0;
      }
  }

  ElectronBlock e{"electrons", ChargedEuler{Euler{gamma}, Real(-1), Real(-1)}, Ue, bc};
  IonBlock ion{"ions", ChargedEulerIsothermal{Real(0.5), Real(0.1), Real(1)}, Ui, bc};
  CoupledSystem system{e, ion};

  // Poisson de systeme : f = q_e n_e + q_i n_i = -n_e + n_i.
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};
  SystemCoupler sim(system, geom, ba, bc, charge);

  // second membre de charge : verifie cellule par cellule que f = q_e n_e + q_i n_i.
  MultiFab rhs(ba, dm, 1, 0);
  charge(system, rhs);
  {
    const ConstArray4 r = rhs.fab(0).const_array();
    const ConstArray4 ue = Ue.fab(0).const_array();
    const ConstArray4 ui = Ui.fab(0).const_array();
    Real maxerr = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const Real expect = Real(-1) * ue(i, j, 0) + Real(1) * ui(i, j, 0);
        const Real d = std::fabs(r(i, j, 0) - expect);
        if (d > maxerr) maxerr = d;
      }
    chk(maxerr < Real(1e-12), "charge_rhs_matches_q_n");
    chk(norm_inf(rhs) > Real(0.9) * eps, "charge_rhs_nonzero");
  }

  const Real me0 = sum(Ue, 0), mi0 = sum(Ui, 0);
  const Real dt = Real(0.001);
  for (int s = 0; s < 8; ++s) sim.step(dt);

  // conservation de la masse par espece (continuite : flux[0] = quantite de mouvement,
  // source[0] = 0 -> integrale conservee sous CL periodiques).
  chk(std::fabs(sum(Ue, 0) - me0) < Real(1e-10), "electron_mass_conserved");
  chk(std::fabs(sum(Ui, 0) - mi0) < Real(1e-10), "ion_mass_conserved");
  // stabilite : densites positives, rien n'a diverge (NaN -> la comparaison echoue).
  chk(norm_inf(Ue) < Real(10) && sum(Ue, 0) > Real(0), "electron_bounded");
  chk(norm_inf(Ui) < Real(10) && sum(Ui, 0) > Real(0), "ion_bounded");

  if (fails == 0) std::printf("OK test_two_species_euler\n");
  return fails == 0 ? 0 : 1;
}
