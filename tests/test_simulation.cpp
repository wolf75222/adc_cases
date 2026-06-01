// Composition multi-especes a l'EXECUTION (TODO 3) : Simulation.add_species(model,...).
//
// Partie A : deux especes de DERIVE (diocotron) ajoutees a la volee -> version runtime
//   de "diocotron a ions mobiles". Masse conservee par espece, Poisson de systeme non nul.
// Partie B : especes HETEROGENES ajoutees au runtime - electrons Euler (4 var) + ions
//   isothermes (3 var) - partageant un meme Poisson. C'est le cas canonique compose
//   espece par espece (et non fige a la compilation comme MultiSpeciesSolver).

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

  const int n = 32;
  const double L = 1.0, k = 2 * M_PI / L, dx = L / n;

  // --- Partie A : deux especes de derive (diocotron) ---
  {
    SimulationConfig cfg;
    cfg.n = n; cfg.L = L; cfg.B0 = 1.0;
    Simulation sim(cfg);
    sim.add_species("electrons", "diocotron", -1.0);
    sim.add_species("ions", "diocotron", +1.0);
    chk(sim.n_species() == 2, "drift_two_species");

    std::vector<double> ne(n * n), ni(n * n, 1.0);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        ne[j * n + i] = 1.0 + 0.1 * std::cos(k * (i + 0.5) * dx);
    sim.set_density("electrons", ne);
    sim.set_density("ions", ni);

    sim.solve_fields();
    const auto phi = sim.potential();
    double phimax = 0;
    for (double v : phi) phimax = std::fmax(phimax, std::fabs(v));
    chk(phimax > 1e-6, "drift_potential_nonzero");

    const double me0 = sim.mass("electrons"), mi0 = sim.mass("ions");
    sim.advance(0.002, 10);
    chk(std::fabs(sim.mass("electrons") - me0) < 1e-10, "drift_electron_mass");
    chk(std::fabs(sim.mass("ions") - mi0) < 1e-10, "drift_ion_mass");
  }

  // --- Partie B : especes heterogenes (Euler 4 var + isotherme 3 var) ---
  {
    SimulationConfig cfg;
    cfg.n = n; cfg.L = L; cfg.gamma = 1.4; cfg.cs2 = 0.5;
    Simulation sim(cfg);
    sim.add_species("electrons", "electron_euler", -1.0);  // 4 variables
    sim.add_species("ions", "ion_isothermal", +1.0);       // 3 variables
    chk(sim.n_species() == 2, "hetero_two_species");

    std::vector<double> ne(n * n), ni(n * n, 1.0);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        ne[j * n + i] = 1.0 + 0.01 * std::cos(k * (i + 0.5) * dx);
    sim.set_density("electrons", ne);  // pose aussi E (Euler) au repos
    sim.set_density("ions", ni);       // pose aussi qte de mouvement nulle

    const double me0 = sim.mass("electrons"), mi0 = sim.mass("ions");
    sim.advance(0.001, 6);

    // masse conservee par espece (continuite) malgre des tailles d'etat differentes.
    chk(std::fabs(sim.mass("electrons") - me0) < 1e-10, "hetero_electron_mass");
    chk(std::fabs(sim.mass("ions") - mi0) < 1e-10, "hetero_ion_mass");
    chk(std::isfinite(sim.mass("electrons")) && sim.mass("electrons") > 0, "hetero_finite");
    chk(static_cast<int>(sim.density("electrons").size()) == n * n, "hetero_density_size");
  }

  if (fails == 0) std::printf("OK test_simulation\n");
  return fails == 0 ? 0 : 1;
}
