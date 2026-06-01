#include <adc/solver/simulation.hpp>

#include <adc/coupling/elliptic_rhs.hpp>  // add_scaled_component
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/model/charged_fluid.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/spatial_operator.hpp>  // assemble_rhs

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
    // fermeture d'avancee (SSPRK2), figee au modele compile a l'ajout de l'espece.
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

  // SSPRK2 generique sur un modele compile (aux gele sur le pas). Type-erased au niveau
  // de la fermeture, mais le noyau (assemble_rhs) est compile pour le modele concret.
  template <class Model>
  void ssprk2(const Model& model, MultiFab& U, Real dt) {
    MultiFab R(ba, dm, Model::n_vars, 0);
    fill_ghosts(U, dom, bc);
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);
    fill_ghosts(U1, dom, bc);
    assemble_rhs<Minmod, RusanovFlux>(model, U1, aux, geom, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
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

void Simulation::add_species(const std::string& name, const std::string& model,
                             double charge) {
  using Kind = Impl::Kind;
  Impl* P = p_.get();
  int ncomp = 1;
  Kind kind = Kind::Diocotron;
  std::function<void(MultiFab&, Real)> advance;

  if (model == "diocotron") {
    ncomp = 1; kind = Kind::Diocotron;
    const Diocotron m{Real(P->cfg.B0), Real(1), Real(1)};
    advance = [P, m](MultiFab& U, Real dt) { P->ssprk2(m, U, dt); };
  } else if (model == "electron_euler") {
    ncomp = 4; kind = Kind::Euler;
    const ChargedEuler m{Euler{Real(P->cfg.gamma)}, Real(charge), Real(charge)};
    advance = [P, m](MultiFab& U, Real dt) { P->ssprk2(m, U, dt); };
  } else if (model == "ion_isothermal") {
    ncomp = 3; kind = Kind::Isothermal;
    const ChargedEulerIsothermal m{Real(P->cfg.cs2), Real(charge), Real(charge)};
    advance = [P, m](MultiFab& U, Real dt) { P->ssprk2(m, U, dt); };
  } else {
    throw std::runtime_error("Simulation::add_species : modele inconnu '" + model +
                             "' (diocotron|electron_euler|ion_isothermal)");
  }

  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), charge, kind,
                                std::move(advance)});
  P->sp.back().U.set_val(Real(0));
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
