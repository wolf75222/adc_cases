#pragma once

#include <memory>
#include <vector>

// Facade compilee MULTI-ESPECES (TODO 3). Instancie un CoupledSystem + SystemCoupler
// du coeur derriere un ABI stable (PIMPL), comme DiocotronSolver / TwoFluidAPSolver.
//
// Cas concret expose : electrons Euler complet + ions Euler isothermes + Poisson de
// systeme (densite de charge q_e n_e + q_i n_i). Python COMPOSE en choisissant la
// config ; aucun callback Python dans le hot path (la physique est en C++ compile,
// schemas et politiques fixes a la compilation). La composition entierement generique
// (vector<SpeciesConfig> arbitraires) reste un cap ulterieur : ici un systeme deux
// fluides representatif valide la chaine Python -> CoupledSystem -> SystemCoupler.

namespace adc {

struct MultiSpeciesConfig {
  int n = 64;             // cellules par direction
  double L = 1.0;         // taille du domaine (periodique)
  double gamma = 1.4;     // electrons : Euler complet
  double cs2_i = 0.5;     // ions : vitesse du son isotherme au carre
  double qom_e = -1.0;    // q/m electrons (force electrostatique)
  double qom_i = 0.1;     // q/m ions (plus lourds)
  double q_e = -1.0;      // charge electrons (Poisson)
  double q_i = 1.0;       // charge ions
  double eps = 0.01;      // amplitude de la perturbation initiale de densite electronique
};

class MultiSpeciesSolver {
 public:
  explicit MultiSpeciesSolver(const MultiSpeciesConfig& cfg);
  ~MultiSpeciesSolver();
  MultiSpeciesSolver(MultiSpeciesSolver&&) noexcept;
  MultiSpeciesSolver& operator=(MultiSpeciesSolver&&) noexcept;

  void step(double dt);
  void advance(double dt, int nsteps);

  int nx() const;
  double time() const;
  double mass_e() const;       // somme de n_e (cellules valides)
  double mass_i() const;       // somme de n_i
  double max_charge() const;   // max |q_e n_e + q_i n_i|
  std::vector<double> density_e() const;   // n x n, row-major
  std::vector<double> density_i() const;
  std::vector<double> potential() const;   // phi, n x n row-major

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
