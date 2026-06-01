// Facade compilee MultiSpeciesSolver (TODO 3) : la chaine que Python pilotera.
// Compose un systeme deux fluides (electrons Euler + ions isothermes + Poisson) via
// CoupledSystem + SystemCoupler derriere un ABI stable, et l'avance. On verifie la
// conservation de la masse par espece, un potentiel exploitable, et la stabilite.

#include <adc/solver/multispecies_solver.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  MultiSpeciesConfig cfg;
  cfg.n = 32;
  cfg.eps = 0.02;
  MultiSpeciesSolver sim(cfg);

  chk(sim.nx() == 32, "nx");
  chk(static_cast<int>(sim.density_e().size()) == 32 * 32, "density_e_size");
  chk(static_cast<int>(sim.potential().size()) == 32 * 32, "potential_size");
  chk(sim.max_charge() > 0.0, "charge_nonzero");  // perturbation -> separation de charge

  const double me0 = sim.mass_e(), mi0 = sim.mass_i();
  sim.advance(0.001, 8);

  chk(std::fabs(sim.mass_e() - me0) < 1e-9, "electron_mass_conserved");
  chk(std::fabs(sim.mass_i() - mi0) < 1e-9, "ion_mass_conserved");
  chk(sim.time() > 0.0, "time_advanced");
  // stabilite : densites finies et positives en moyenne.
  chk(std::isfinite(sim.mass_e()) && sim.mass_e() > 0.0, "electron_finite");
  chk(std::isfinite(sim.mass_i()) && sim.mass_i() > 0.0, "ion_finite");

  if (fails == 0) std::printf("OK test_multispecies_solver\n");
  return fails == 0 ? 0 : 1;
}
