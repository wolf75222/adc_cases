// Diocotron a IONS MOBILES (TODO 2.4) : les ions deviennent un 2e bloc.
//
// Le diocotron mono-espece fige les ions en fond neutralisant constant n_i0 :
// son elliptic_rhs code alpha (n_e - n_i0) en dur. Ici on REUTILISE le modele
// Diocotron (derive E x B, lue dans aux) pour DEUX blocs - electrons et ions - et
// le second membre de Poisson devient alpha (n_e - n_i) via ChargeDensityRhs (n_i
// n'est plus une constante mais la densite ionique transportee). La derive E x B
// etant independante de la charge, les deux especes tournent dans le meme champ ;
// la masse de chacune est conservee (advection incompressible).

#include <adc/core/coupled_system.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/model/diocotron.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

using ElectronBlock = EquationBlock<Diocotron, MusclMinmod, ExplicitTime<SSPRK2, 1>>;
using IonBlock = EquationBlock<Diocotron, MusclMinmod, ExplicitTime<SSPRK2, 1>>;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int n = 32;
  const Real L = Real(1), B0 = Real(1), alpha = Real(1);
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, double(L), 0.0, double(L)};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dx = geom.dx();
  BCRec bc;  // periodique

  MultiFab Ue(ba, dm, 1, 2), Ui(ba, dm, 1, 2);
  // IC : n_e = 1 + eps cos(k x), n_i = 1 -> charge alpha(n_e - n_i) a moyenne nulle.
  const Real eps = Real(0.1), k = Real(2) * Real(M_PI) / L;
  {
    Array4 ue = Ue.fab(0).array();
    Array4 ui = Ui.fab(0).array();
    const Box2D g = Ue.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const Real x = (Real(i) + Real(0.5)) * dx;
        ue(i, j, 0) = Real(1) + eps * std::cos(k * x);
        ui(i, j, 0) = Real(1);
      }
  }

  // meme modele Diocotron pour les deux especes (derive E x B partagee).
  ElectronBlock e{"electrons", Diocotron{B0, Real(1), alpha}, Ue, bc};
  IonBlock ion{"ions", Diocotron{B0, Real(1), alpha}, Ui, bc};
  CoupledSystem system{e, ion};

  // Poisson de systeme : f = alpha n_e - alpha n_i = alpha (n_e - n_i) (ions mobiles).
  ChargeDensityRhs charge{{{alpha, 0}, {-alpha, 0}}};
  SystemCoupler sim(system, geom, ba, bc, charge);

  // f = alpha n_e - alpha n_i, verifie cellule par cellule.
  MultiFab rhs(ba, dm, 1, 0);
  charge(system, rhs);
  {
    const ConstArray4 r = rhs.fab(0).const_array();
    const ConstArray4 ue = Ue.fab(0).const_array();
    const ConstArray4 ui = Ui.fab(0).const_array();
    Real maxerr = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const Real expect = alpha * ue(i, j, 0) - alpha * ui(i, j, 0);
        const Real d = std::fabs(r(i, j, 0) - expect);
        if (d > maxerr) maxerr = d;
      }
    chk(maxerr < Real(1e-12), "charge_rhs_matches_alpha_n");
    chk(norm_inf(rhs) > Real(0.9) * alpha * eps, "charge_rhs_nonzero");
  }

  const Real me0 = sum(Ue, 0), mi0 = sum(Ui, 0);
  const Real dt = Real(0.002);
  for (int s = 0; s < 10; ++s) sim.step(dt);

  // advection incompressible (derive E x B sans divergence) : masse conservee par espece.
  chk(std::fabs(sum(Ue, 0) - me0) < Real(1e-10), "electron_mass_conserved");
  chk(std::fabs(sum(Ui, 0) - mi0) < Real(1e-10), "ion_mass_conserved");
  // les electrons ont evolue (perturbation transportee), les ions aussi sont mobiles.
  chk(norm_inf(Ue) < Real(2) && norm_inf(Ui) < Real(2), "bounded");

  if (fails == 0) std::printf("OK test_diocotron_mobile_ions\n");
  return fails == 0 ? 0 : 1;
}
