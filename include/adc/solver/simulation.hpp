#pragma once

#include <memory>
#include <string>
#include <vector>

// Composition MULTI-ESPECES a l'EXECUTION (TODO 3, version "Simulation").
//
// La facade MultiSpeciesSolver fige UN systeme deux fluides a la compilation. Ici
// l'utilisateur COMPOSE a l'execution : il ajoute autant d'especes qu'il veut, chacune
// avec sa charge, toutes partageant un meme Poisson (densite de charge Sum_s q_s n_s)
// et un meme champ de derive E x B. C'est l'esprit `sim.add_equation(...)` du TODO,
// borne ici aux especes de type DERIVE (Diocotron, 1 variable -> CI simples) pour rester
// un squelette testable ; la physique (transport, assemblage) reste en C++ compile,
// aucun callback Python dans le hot path. Etendre a Euler / IMEX = meme patron + une CI
// par modele.
//
// Le hot path n'est PAS type-erased cellule par cellule : add_species selectionne un
// modele compile (Diocotron) ; seule la LISTE d'especes est dynamique.

namespace adc {

struct SimulationConfig {
  int n = 64;          // cellules par direction
  double L = 1.0;      // taille du domaine
  double B0 = 1.0;     // champ magnetique (derive E x B partagee)
  bool periodic = true;
};

class Simulation {
 public:
  explicit Simulation(const SimulationConfig& cfg);
  ~Simulation();
  Simulation(Simulation&&) noexcept;
  Simulation& operator=(Simulation&&) noexcept;

  // Ajoute une espece de derive de charge donnee (densite initiale nulle).
  void add_species(const std::string& name, double charge);
  // Fixe la densite d'une espece (tableau n*n row-major).
  void set_density(const std::string& name, const std::vector<double>& rho);

  void solve_fields();           // resout Poisson (Sum_s q_s n_s) + aux = grad phi
  void step(double dt);          // solve_fields puis avance chaque espece (SSPRK2)
  void advance(double dt, int nsteps);

  int nx() const;
  double time() const;
  int n_species() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  // n*n row-major
  std::vector<double> potential() const;                       // phi, n*n row-major

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
