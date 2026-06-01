#include <adc/solver/multispecies_solver.hpp>

#include <adc/core/coupled_system.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/model/charged_fluid.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>

namespace adc {

namespace {
using EBlk = EquationBlock<ChargedEuler, MusclMinmod, ExplicitTime<SSPRK2, 1>>;
using IBlk = EquationBlock<ChargedEulerIsothermal, MusclMinmod, ExplicitTime<SSPRK2, 1>>;
using Sys = CoupledSystem<EBlk, IBlk>;
using SysCoupler = SystemCoupler<Sys, ChargeDensityRhs>;
}  // namespace

struct MultiSpeciesSolver::Impl {
  MultiSpeciesConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;          // periodique partout
  MultiFab Ue, Ui;   // detenus ici (adresse stable) ; les blocs du systeme pointent dessus
  SysCoupler sim;
  double t = 0;

  // Construit le systeme dont les blocs referencent Ue/Ui (deja construits, adresse
  // stable). Le SystemCoupler deplace le systeme mais les pointeurs d'etat survivent.
  static Sys make_system(const MultiSpeciesConfig& c, MultiFab& Ue, MultiFab& Ui) {
    EBlk e{"electrons",
           ChargedEuler{Euler{Real(c.gamma)}, Real(c.qom_e), Real(c.q_e)}, Ue, BCRec{}};
    IBlk i{"ions",
           ChargedEulerIsothermal{Real(c.cs2_i), Real(c.qom_i), Real(c.q_i)}, Ui, BCRec{}};
    return Sys{e, i};
  }
  static ChargeDensityRhs make_charge(const MultiSpeciesConfig& c) {
    return ChargeDensityRhs{{{Real(c.q_e), 0}, {Real(c.q_i), 0}}};
  }

  explicit Impl(const MultiSpeciesConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        Ue(ba, dm, 4, 2),
        Ui(ba, dm, 3, 2),
        sim(make_system(c, Ue, Ui), geom, ba, bc, make_charge(c)) {
    fill_ic();
    sim.solve_fields();  // phi valide avant tout pas (potential() utilisable)
  }

  void fill_ic() {
    const Real dx = geom.dx();
    const Real k = Real(2) * Real(M_PI) / Real(cfg.L);
    const Real g = Real(cfg.gamma), eps = Real(cfg.eps);
    Array4 ue = Ue.fab(0).array();
    Array4 ui = Ui.fab(0).array();
    const Box2D gb = Ue.fab(0).grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
        const Real x = (Real(i) + Real(0.5)) * dx;
        const Real ne = Real(1) + eps * std::cos(k * x);
        ue(i, j, 0) = ne; ue(i, j, 1) = 0; ue(i, j, 2) = 0;
        ue(i, j, 3) = ne / (g - Real(1));  // E = p/(g-1), p = ne, KE = 0
        ui(i, j, 0) = Real(1); ui(i, j, 1) = 0; ui(i, j, 2) = 0;
      }
  }

  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    const ConstArray4 a = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(a(i, j, 0));
    return out;
  }

  double max_charge() const {
    device_fence();
    const ConstArray4 fe = Ue.fab(0).const_array(), fi = Ui.fab(0).const_array();
    const Box2D v = Ue.box(0);
    const double qe = cfg.q_e, qi = cfg.q_i;
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m = std::fmax(m, std::fabs(qe * fe(i, j, 0) + qi * fi(i, j, 0)));
    return m;
  }
};

MultiSpeciesSolver::MultiSpeciesSolver(const MultiSpeciesConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
MultiSpeciesSolver::~MultiSpeciesSolver() = default;
MultiSpeciesSolver::MultiSpeciesSolver(MultiSpeciesSolver&&) noexcept = default;
MultiSpeciesSolver& MultiSpeciesSolver::operator=(MultiSpeciesSolver&&) noexcept = default;

void MultiSpeciesSolver::step(double dt) {
  p_->sim.step(dt);  // blocs explicites : avances par le coeur
  p_->t += dt;
}
void MultiSpeciesSolver::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}

int MultiSpeciesSolver::nx() const { return p_->cfg.n; }
double MultiSpeciesSolver::time() const { return p_->t; }
double MultiSpeciesSolver::mass_e() const { return sum(p_->Ue, 0); }
double MultiSpeciesSolver::mass_i() const { return sum(p_->Ui, 0); }
double MultiSpeciesSolver::max_charge() const { return p_->max_charge(); }
std::vector<double> MultiSpeciesSolver::density_e() const { return p_->copy_comp0(p_->Ue); }
std::vector<double> MultiSpeciesSolver::density_i() const { return p_->copy_comp0(p_->Ui); }
std::vector<double> MultiSpeciesSolver::potential() const {
  return p_->copy_comp0(p_->sim.phi());
}

}  // namespace adc
