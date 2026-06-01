#include <adc/solver/simulation.hpp>

#include <adc/coupling/elliptic_rhs.hpp>  // add_scaled_component
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/spatial_operator.hpp>  // assemble_rhs

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // saxpy, lincomb, sum
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // fill_ghosts

#include <stdexcept>

namespace adc {

struct Simulation::Impl {
  struct Species {
    std::string name;
    MultiFab U;       // 1 variable (densite)
    double charge;
  };

  SimulationConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;          // CL du potentiel et des especes (periodique par defaut)
  Box2D dom;
  Periodicity per;
  GeometricMG mg;
  MultiFab aux;      // (phi, grad phi) partage
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
    BCRec b;  // periodique par defaut ; sinon outflow (Foextrap) au bord
    if (!c.periodic) {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    }
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

  // SSPRK2 d'une espece de derive (modele Diocotron), aux gele sur le pas.
  void advance_species(Species& s, Real dt) {
    const Diocotron model{Real(cfg.B0), Real(1), Real(1)};
    MultiFab R(ba, dm, 1, 0);
    MultiFab& U = s.U;
    fill_ghosts(U, dom, bc);
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);
    fill_ghosts(U1, dom, bc);
    assemble_rhs<Minmod, RusanovFlux>(model, U1, aux, geom, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
  }

  std::vector<double> copy_density(const Species& s) const {
    device_fence();
    const ConstArray4 u = s.U.fab(0).const_array();
    const Box2D v = s.U.box(0);
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

void Simulation::add_species(const std::string& name, double charge) {
  p_->sp.push_back(Impl::Species{name, MultiFab(p_->ba, p_->dm, 1, 2), charge});
  p_->sp.back().U.set_val(Real(0));
}

void Simulation::set_density(const std::string& name,
                             const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const int n = p_->cfg.n;
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("Simulation::set_density : taille != n*n");
  Array4 u = s.U.fab(0).array();
  const Box2D v = s.U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      u(i, j, 0) = rho[static_cast<std::size_t>(j) * n + i];
}

void Simulation::solve_fields() { p_->solve_fields(); }

void Simulation::step(double dt) {
  p_->solve_fields();
  for (auto& s : p_->sp) p_->advance_species(s, Real(dt));
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
  return p_->copy_density(p_->find(name));
}
std::vector<double> Simulation::potential() const {
  p_->mg.phi();  // phi a jour apres solve_fields
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
