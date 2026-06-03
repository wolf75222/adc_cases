// ABI C de l'integrateur deux-fluides AP, chargeable par ctypes depuis run.py.
//
// L'AP deux-fluides a quitte le coeur adc_cpp (c'est un SCENARIO, pas une brique generique).
// On le compile ICI, dans adc_cases, en une bibliotheque partagee chargee a la volee
// (cf. run.py) : meme principe que le JIT du DSL (adc_cpp/python/tests/test_dsl_jitlib.py),
// mais sans pybind11 cote cas. Toute la physique vit dans two_fluid_ap.hpp ; ce fichier ne
// fait qu'exposer une poignee de fonctions extern "C" (creer / pas / diagnostics / densites).
//
// GeometricMG : elliptique entierement on-device (lisseur GS rb + V-cycle via for_each_cell),
// donc la facade compile telle quelle pour le GPU. Le backend (serie/OpenMP/Kokkos) est herite
// de la facon dont -I adc_cpp/include + les flags sont passes a la compilation.

#include "two_fluid_ap.hpp"

#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using adc::Box2D;
using adc::ConstArray4;
using adc::GeometricMG;
using adc::MultiFab;
using adc::Real;
using adc::TwoFluidAP2D;

// Solveur concret + parametres de pas (stabilize) caches derriere un handle opaque.
struct Solver {
  TwoFluidAP2D<GeometricMG> d;
  bool stabilize;

  Solver(int n, double L, double cse2, double csi2, double omega_pe, double omega_pi,
         bool stab, double eps, bool upwind, double omega_ce, double omega_ci)
      : d(n, L, cse2, csi2, omega_pe, omega_pi), stabilize(stab) {
    d.upwind_continuity = upwind;
    d.wce = omega_ce;
    d.wci = omega_ci;
    d.init(eps);
  }

  void copy_comp(const MultiFab& mf, double* out) const {
    adc::device_fence();  // GPU : barriere avant lecture hote (memoire unifiee)
    const ConstArray4 a = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::size_t k = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out[k++] = a(i, j, 0);
  }

  double max_charge() const {
    adc::device_fence();
    const ConstArray4 fe = d.e.fab(0).const_array(), fi = d.ion.fab(0).const_array();
    const Box2D v = d.e.box(0);
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m = std::fmax(m, std::fabs(fi(i, j, 0) - fe(i, j, 0)));
    return m;
  }
  double max_dev() const {
    adc::device_fence();
    const ConstArray4 fe = d.e.fab(0).const_array();
    const Box2D v = d.e.box(0);
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m = std::fmax(m, std::fabs(fe(i, j, 0) - 1.0));
    return m;
  }
};

}  // namespace

extern "C" {

// Cree un solveur (les champs correspondent a l'ancien TwoFluidAPConfig). Renvoie un handle.
void* tfap_create(int n, double L, double cse2, double csi2, double omega_pe, double omega_pi,
                  int stabilize, double eps, int upwind_continuity, double omega_ce,
                  double omega_ci) {
  return new Solver(n, L, cse2, csi2, omega_pe, omega_pi, stabilize != 0, eps,
                    upwind_continuity != 0, omega_ce, omega_ci);
}

void tfap_destroy(void* h) { delete static_cast<Solver*>(h); }

void tfap_step(void* h, double dt) {
  Solver* s = static_cast<Solver*>(h);
  s->d.step(dt, s->stabilize);
}
void tfap_advance(void* h, double dt, int nsteps) {
  Solver* s = static_cast<Solver*>(h);
  for (int k = 0; k < nsteps; ++k) s->d.step(dt, s->stabilize);
}

int tfap_nx(void* h) { return static_cast<Solver*>(h)->d.n; }
double tfap_mass_e(void* h) { return adc::sum(static_cast<Solver*>(h)->d.e, 0); }
double tfap_mass_i(void* h) { return adc::sum(static_cast<Solver*>(h)->d.ion, 0); }
double tfap_max_charge(void* h) { return static_cast<Solver*>(h)->max_charge(); }
double tfap_max_dev(void* h) { return static_cast<Solver*>(h)->max_dev(); }

// densites n_e / n_i ecrites dans out (n*n, row-major) ; out doit avoir nx*nx doubles.
void tfap_density_e(void* h, double* out) {
  Solver* s = static_cast<Solver*>(h);
  s->copy_comp(s->d.e, out);
}
void tfap_density_i(void* h, double* out) {
  Solver* s = static_cast<Solver*>(h);
  s->copy_comp(s->d.ion, out);
}

}  // extern "C"
