// Composition multi-especes a l'EXECUTION (TODO 3) : Simulation.add_species(...).
// Version runtime de "diocotron a ions mobiles", composee espece par espece (et non
// figee a la compilation). On ajoute electrons + ions, on fixe leurs densites, on
// avance, et on verifie la conservation de la masse PAR espece et un potentiel
// exploitable (Poisson de systeme Sum_s q_s n_s sur des especes ajoutees a la volee).

#include <adc/solver/simulation.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  SimulationConfig cfg;
  cfg.n = 32;
  cfg.L = 1.0;
  cfg.B0 = 1.0;
  Simulation sim(cfg);

  sim.add_species("electrons", -1.0);  // charge -1
  sim.add_species("ions", +1.0);       // charge +1
  chk(sim.n_species() == 2, "two_species_added");

  // CI : n_e = 1 + eps cos(k x), n_i = 1 -> charge a moyenne nulle.
  const int n = cfg.n;
  const double eps = 0.1, k = 2 * M_PI / cfg.L, dx = cfg.L / n;
  std::vector<double> ne(n * n), ni(n * n, 1.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      ne[j * n + i] = 1.0 + eps * std::cos(k * (i + 0.5) * dx);
  sim.set_density("electrons", ne);
  sim.set_density("ions", ni);

  sim.solve_fields();
  // potentiel non trivial (separation de charge -> Poisson non nul).
  const auto phi = sim.potential();
  double phimax = 0;
  for (double v : phi) phimax = std::fmax(phimax, std::fabs(v));
  chk(static_cast<int>(phi.size()) == n * n, "potential_size");
  chk(phimax > 1e-6, "potential_nonzero");

  const double me0 = sim.mass("electrons"), mi0 = sim.mass("ions");
  sim.advance(0.002, 10);

  // masse conservee par espece (advection E x B incompressible, composee au runtime).
  chk(std::fabs(sim.mass("electrons") - me0) < 1e-10, "electron_mass_conserved");
  chk(std::fabs(sim.mass("ions") - mi0) < 1e-10, "ion_mass_conserved");
  chk(static_cast<int>(sim.density("electrons").size()) == n * n, "density_size");
  chk(sim.time() > 0.0, "time_advanced");

  if (fails == 0) std::printf("OK test_simulation\n");
  return fails == 0 ? 0 : 1;
}
