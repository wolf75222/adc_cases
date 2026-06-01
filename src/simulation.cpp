#include <adc/solver/simulation.hpp>

#include <adc/coupling/elliptic_rhs.hpp>       // add_scaled_component
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/implicit_stepper.hpp>  // backward_euler_source
#include <adc/model/charged_fluid.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/spatial_operator.hpp>    // assemble_rhs, SourceFreeModel

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // saxpy, lincomb, sum
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // fill_ghosts

#include <functional>
#include <stdexcept>

namespace adc {

struct Simulation::Impl {
  enum class Kind { Diocotron, Euler, Isothermal };

  struct Species {
    std::string name;
    MultiFab U;
    double charge;
    Kind kind;
    // fermeture d'avancee figee a l'ajout : modele + schema spatial (Limiter, Flux) +
    // traitement temporel (explicite / IMEX) + sous-pas, tout compile. Type-erased SEULEMENT
    // au niveau de la liste d'especes ; le noyau (assemble_rhs<L,F>) reste compile.
    std::function<void(MultiFab&, Real)> advance;
  };

  SimulationConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;
  Box2D dom;
  Periodicity per;
  GeometricMG mg;
  MultiFab aux;
  std::vector<Species> sp;
  double t = 0;

  explicit Impl(const SimulationConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        bc(make_bc(c)),
        dom(Box2D::from_extents(c.n, c.n)),
        per{c.periodic, c.periodic},
        mg(geom, ba, bc),
        aux(ba, dm, 3, 1) {}

  static BCRec make_bc(const SimulationConfig& c) {
    BCRec b;
    if (!c.periodic) b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  Species& find(const std::string& name) {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("Simulation: espece inconnue '" + name + "'");
  }
  const Species& find(const std::string& name) const {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("Simulation: espece inconnue '" + name + "'");
  }

  // SSPRK2 sur un modele compile, schema spatial (Limiter, Flux) en parametres de template
  // (aux gele sur le pas). Type-erased a la liste d'especes, mais le noyau est compile.
  template <class Limiter, class Flux, class Model>
  void ssprk2(const Model& model, MultiFab& U, Real dt) {
    MultiFab R(ba, dm, Model::n_vars, 0);
    fill_ghosts(U, dom, bc);
    assemble_rhs<Limiter, Flux>(model, U, aux, geom, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);
    fill_ghosts(U1, dom, bc);
    assemble_rhs<Limiter, Flux>(model, U1, aux, geom, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
  }

  // Pas IMEX : transport EXPLICITE (−div F, modele source-free) puis source IMPLICITE
  // (backward-Euler / Newton local). C'est "electrons implicites" cote source raide.
  template <class Limiter, class Flux, class Model>
  void imex_step(const Model& model, MultiFab& U, Real dt) {
    const SourceFreeModel<Model> sf{model};
    MultiFab R(ba, dm, Model::n_vars, 0);
    fill_ghosts(U, dom, bc);
    assemble_rhs<Limiter, Flux>(sf, U, aux, geom, R);
    saxpy(U, dt, R);                              // transport explicite (Euler avant)
    backward_euler_source(model, aux, U, dt);     // source implicite
  }

  // Fermeture pour (Limiter, Flux) fixes : sous-cyclage + explicite/IMEX.
  template <class Limiter, class Flux, class Model>
  std::function<void(MultiFab&, Real)> closure(const Model& m, bool imex, int substeps) {
    Impl* P = this;
    if (imex)
      return [P, m, substeps](MultiFab& U, Real dt) {
        const Real h = dt / static_cast<Real>(substeps);
        for (int s = 0; s < substeps; ++s) P->imex_step<Limiter, Flux>(m, U, h);
      };
    return [P, m, substeps](MultiFab& U, Real dt) {
      const Real h = dt / static_cast<Real>(substeps);
      for (int s = 0; s < substeps; ++s) P->ssprk2<Limiter, Flux>(m, U, h);
    };
  }

  // Dispatch des tags (limiter, flux) -> fermeture compilee. HLLC garde par `requires`
  // (n'est instancie que pour un modele exposant wave_speeds, sinon erreur claire).
  template <class Model>
  std::function<void(MultiFab&, Real)> make_advance(const Model& m, const std::string& lim,
                                                    const std::string& flx, bool imex,
                                                    int substeps) {
    if (flx == "rusanov") {
      if (lim == "none") return closure<NoSlope, RusanovFlux>(m, imex, substeps);
      if (lim == "minmod") return closure<Minmod, RusanovFlux>(m, imex, substeps);
      if (lim == "vanleer") return closure<VanLeer, RusanovFlux>(m, imex, substeps);
      throw std::runtime_error("Simulation: limiter inconnu '" + lim + "'");
    }
    if (flx == "hllc") {
      // HLLC restitue l'onde de contact : exige la pression ET 4 variables (energie en
      // [3]). Le modele isotherme (3 var, sans pression) et diocotron en sont exclus ->
      // erreur claire les renvoyant vers 'rusanov'.
      if constexpr (Model::n_vars == 4 &&
                    requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
        if (lim == "none") return closure<NoSlope, HLLCFlux>(m, imex, substeps);
        if (lim == "minmod") return closure<Minmod, HLLCFlux>(m, imex, substeps);
        if (lim == "vanleer") return closure<VanLeer, HLLCFlux>(m, imex, substeps);
        throw std::runtime_error("Simulation: limiter inconnu '" + lim + "'");
      } else {
        throw std::runtime_error("Simulation: flux 'hllc' exige un modele Euler complet "
                                 "(4 variables + pression) ; ce modele -> 'rusanov'");
      }
    }
    throw std::runtime_error("Simulation: flux inconnu '" + flx + "' (rusanov|hllc)");
  }

  void solve_fields() {
    mg.rhs().set_val(Real(0));
    for (auto& s : sp) add_scaled_component(s.U, Real(s.charge), 0, mg.rhs());
    mg.solve();
    device_fence();
    const Real dx = geom.dx(), dy = geom.dy();
    const ConstArray4 p = mg.phi().fab(0).const_array();
    Array4 a = aux.fab(0).array();
    const Box2D v = aux.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        a(i, j, 0) = p(i, j);
        a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    fill_boundary(aux, dom, per);
  }

  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }
};

Simulation::Simulation(const SimulationConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
Simulation::~Simulation() = default;
Simulation::Simulation(Simulation&&) noexcept = default;
Simulation& Simulation::operator=(Simulation&&) noexcept = default;

void Simulation::add_block(const std::string& name, const std::string& model, double charge,
                           const std::string& limiter, const std::string& flux,
                           const std::string& time, int substeps) {
  using Kind = Impl::Kind;
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("Simulation::add_block : substeps >= 1");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("Simulation::add_block : time 'explicit' | 'imex' (recu '" +
                             time + "')");
  const bool imex = (time == "imex");

  int ncomp = 1;
  Kind kind = Kind::Diocotron;
  std::function<void(MultiFab&, Real)> advance;
  if (model == "diocotron") {
    ncomp = 1; kind = Kind::Diocotron;
    advance = P->make_advance(Diocotron{Real(P->cfg.B0), Real(1), Real(1)}, limiter, flux,
                              imex, substeps);
  } else if (model == "electron_euler") {
    ncomp = 4; kind = Kind::Euler;
    advance = P->make_advance(ChargedEuler{Euler{Real(P->cfg.gamma)}, Real(charge),
                                           Real(charge)}, limiter, flux, imex, substeps);
  } else if (model == "ion_isothermal") {
    ncomp = 3; kind = Kind::Isothermal;
    advance = P->make_advance(ChargedEulerIsothermal{Real(P->cfg.cs2), Real(charge),
                                                     Real(charge)}, limiter, flux, imex,
                              substeps);
  } else {
    throw std::runtime_error("Simulation::add_block : modele inconnu '" + model +
                             "' (diocotron|electron_euler|ion_isothermal)");
  }

  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), charge, kind,
                                std::move(advance)});
  P->sp.back().U.set_val(Real(0));
}

void Simulation::add_species(const std::string& name, const std::string& model,
                             double charge) {
  add_block(name, model, charge);  // minmod + rusanov + explicite + 1 sous-pas
}

void Simulation::set_density(const std::string& name,
                             const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const int n = p_->cfg.n;
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("Simulation::set_density : taille != n*n");
  const Real gm1 = Real(p_->cfg.gamma) - Real(1);
  Array4 u = s.U.fab(0).array();
  const Box2D v = s.U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j) * n + i];
      u(i, j, 0) = r;
      if (s.kind == Impl::Kind::Euler) {
        u(i, j, 1) = 0; u(i, j, 2) = 0;
        u(i, j, 3) = r / gm1;  // E = p/(g-1), p = rho, au repos
      } else if (s.kind == Impl::Kind::Isothermal) {
        u(i, j, 1) = 0; u(i, j, 2) = 0;
      }
    }
}

void Simulation::solve_fields() { p_->solve_fields(); }

void Simulation::step(double dt) {
  p_->solve_fields();
  for (auto& s : p_->sp) s.advance(s.U, Real(dt));
  p_->t += dt;
}
void Simulation::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}

int Simulation::nx() const { return p_->cfg.n; }
double Simulation::time() const { return p_->t; }
int Simulation::n_species() const { return static_cast<int>(p_->sp.size()); }
double Simulation::mass(const std::string& name) const {
  return sum(p_->find(name).U, 0);
}
std::vector<double> Simulation::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> Simulation::potential() const {
  device_fence();
  const ConstArray4 ph = p_->mg.phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

}  // namespace adc
