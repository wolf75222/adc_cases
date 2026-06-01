// Pilote MINCE de composition multi-especes a l'execution (TODO 3), sur la facade
// Simulation (adc_cases::solver). Compose electrons + ions a la volee, partageant un
// Poisson de systeme, et ecrit : un CSV de diagnostics (masse par espece, |charge| max)
// + des instantanes de densite electronique. Toute la physique est dans libadc.
//
// Args : [out_dir] [n] [nsteps] [model_e: diocotron|electron_euler] [charge_e]
// Defaut : diocotron a ions mobiles (electrons + ions de derive), n=128, 400 pas.

#include <adc/solver/simulation.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace adc;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "multispecies_out";
  const int n = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 400;
  const std::string model_e = (argc > 4) ? argv[4] : "diocotron";
  const double qe = (argc > 5) ? std::atof(argv[5]) : -1.0;
  const std::string model_i = (model_e == "diocotron") ? "diocotron" : "ion_isothermal";
  std::filesystem::create_directories(out);

  SimulationConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.B0 = 1.0;
  Simulation sim(cfg);
  sim.add_species("electrons", model_e, qe);
  sim.add_species("ions", model_i, -qe);

  // CI quasi-neutre : n_e perturbee, n_i = 1 -> charge a moyenne nulle (Poisson periodique).
  const double k = 2 * M_PI / cfg.L, dx = cfg.L / n;
  std::vector<double> ne(n * n), ni(n * n, 1.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      ne[j * n + i] =
          1.0 + 0.1 * std::cos(k * (i + 0.5) * dx) * std::cos(k * (j + 0.5) * dx);
  sim.set_density("electrons", ne);
  sim.set_density("ions", ni);
  sim.solve_fields();

  auto dump = [&](int frame) {
    const auto d = sim.density("electrons");  // n x n row-major
    char name[64];
    std::snprintf(name, sizeof(name), "/ne_%04d.txt", frame);
    std::ofstream f(out + name);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) f << d[j * n + i] << (i + 1 < n ? ' ' : '\n');
  };

  std::ofstream diag(out + "/multispecies.csv");
  diag << "# multispecies (facade Simulation) n=" << n << " model_e=" << model_e
       << " model_i=" << model_i << "\n";
  diag << "t,mass_e,mass_i,charge_e\n";

  const int snap_every = std::max(1, nsteps / 20);
  int frame = 0;
  const double me0 = sim.mass("electrons"), mi0 = sim.mass("ions");
  std::printf("multispecies n=%d nsteps=%d especes=%d (%s + %s)\n", n, nsteps,
              sim.n_species(), model_e.c_str(), model_i.c_str());

  for (int step = 0; step <= nsteps; ++step) {
    diag << sim.time() << ',' << sim.mass("electrons") << ',' << sim.mass("ions") << ','
         << qe * sim.mass("electrons") << '\n';
    if (step % snap_every == 0) {
      dump(frame++);
      std::printf("  s=%5d t=%7.3f  d(masse_e)=%.2e  d(masse_i)=%.2e\n", step, sim.time(),
                  sim.mass("electrons") - me0, sim.mass("ions") - mi0);
    }
    if (step == nsteps) break;
    sim.step(0.002);
  }
  std::printf("ecrit %s/multispecies.csv + %d instantanes (d masse_e=%.2e)\n", out.c_str(),
              frame, sim.mass("electrons") - me0);
  return 0;
}
