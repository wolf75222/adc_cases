#pragma once

#include <memory>
#include <string>
#include <vector>

// Composition MULTI-ESPECES a l'EXECUTION (TODO 3, version "Simulation").
//
// La facade MultiSpeciesSolver fige UN systeme deux fluides a la compilation. Ici
// l'utilisateur COMPOSE a l'execution : il ajoute autant d'especes qu'il veut, chacune
// avec son MODELE et sa charge, toutes partageant un meme Poisson (densite de charge
// Sum_s q_s n_s) et un meme champ E x B / E. C'est l'esprit `sim.add_equation(name,
// model, ...)` du TODO. Modeles disponibles (tag) :
//   "diocotron"       : derive E x B, 1 variable ;
//   "electron_euler"  : Euler complet + force electrostatique, 4 variables ;
//   "ion_isothermal"  : Euler isotherme + force electrostatique, 3 variables.
// La physique (transport, assemblage) reste en C++ compile : add_species selectionne un
// modele compile et fige une fermeture d'avancee (SSPRK2) ; seule la LISTE d'especes
// est dynamique, jamais le noyau cellule par cellule. Aucun callback Python dans le hot
// path. Especes heterogenes (1/3/4 variables) dans une meme Simulation : la densite est
// toujours la composante 0, lue par le Poisson de systeme.

namespace adc {

struct SimulationConfig {
  int n = 64;          // cellules par direction
  double L = 1.0;      // taille du domaine
  double B0 = 1.0;     // champ magnetique (derive E x B, especes "diocotron")
  double gamma = 1.4;  // adiabatique (especes "electron_euler")
  double cs2 = 0.5;    // vitesse du son^2 isotherme (especes "ion_isothermal")
  bool periodic = true;
};

class Simulation {
 public:
  explicit Simulation(const SimulationConfig& cfg);
  ~Simulation();
  Simulation(Simulation&&) noexcept;
  Simulation& operator=(Simulation&&) noexcept;

  // Ajoute un BLOC d'equation (= espece) en COMPOSANT, comme le voulait le tuteur :
  //   model    : "diocotron" | "electron_euler" | "ion_isothermal"
  //   charge   : signe pour le Poisson de systeme (et q/m de la force, masse unite)
  //   limiter  : reconstruction spatiale "none" | "minmod" | "vanleer"
  //   flux     : flux numerique "rusanov" | "hllc"  (hllc exige un modele Euler-like)
  //   time     : "explicit" (SSPRK2) | "imex" (transport explicite + source implicite)
  //   substeps : sous-pas par bloc (10 electrons : 1 ion). Chaque bloc choisit SON schema.
  // C'est le niveau d'assemblage : Python dit QUOI composer, le C++ compile fait le calcul.
  void add_block(const std::string& name, const std::string& model, double charge,
                 const std::string& limiter = "minmod",
                 const std::string& flux = "rusanov",
                 const std::string& time = "explicit", int substeps = 1);
  // Raccourci historique : minmod + rusanov + explicite + 1 sous-pas.
  void add_species(const std::string& name, const std::string& model, double charge);
  // Fixe la densite (composante 0) d'une espece, tableau n*n row-major ; les autres
  // composantes (quantite de mouvement, energie) sont posees a l'equilibre au repos.
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
