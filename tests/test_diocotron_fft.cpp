// Choix du solveur elliptique a l'execution (use_fft) dans la facade diocotron, comme
// pour Euler-Poisson. Sur le cas Band (periodique, n puissance de 2), la multigrille et
// la FFT spectrale resolvent le MEME Poisson : a dt fixe, les trajectoires doivent
// coincider (a la tolerance du V-cycle pres) et conserver la masse.

#include <adc/solver/diocotron_solver.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  DiocotronConfig cm;
  cm.n = 64;
  cm.ic = DiocotronIC::Band;
  cm.band_mode = 2;
  cm.use_fft = false;  // multigrille
  DiocotronConfig cf = cm;
  cf.use_fft = true;  // FFT spectrale

  DiocotronSolver sm(cm), sf(cf);
  const double m0 = sm.mass();
  const double dt = 0.2 * sm.dx();  // dt FIXE : isole la difference de solveur
  for (int i = 0; i < 40; ++i) {
    sm.step(dt);
    sf.step(dt);
  }

  const auto rm = sm.density(), rf = sf.density();
  chk(rm.size() == rf.size(), "same_shape");
  double maxdiff = 0;
  bool finite = true;
  for (std::size_t k = 0; k < rm.size(); ++k) {
    maxdiff = std::max(maxdiff, std::fabs(rm[k] - rf[k]));
    if (!std::isfinite(rf[k])) finite = false;
  }
  std::printf("  MG vs FFT : max|drho|=%.2e  massMG=%.8f  massFFT=%.8f\n", maxdiff,
              sm.mass(), sf.mass());
  chk(finite, "fft_finite");
  chk(std::fabs(sm.mass() - m0) < 1e-6, "mg_mass_conserved");
  chk(std::fabs(sf.mass() - m0) < 1e-6, "fft_mass_conserved");
  chk(maxdiff < 1e-2, "mg_and_fft_agree");  // meme physique, solveur different

  if (fails == 0) std::printf("OK test_diocotron_fft\n");
  return fails == 0 ? 0 : 1;
}
