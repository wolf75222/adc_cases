"""Teste le module Python `adc` (binding de la facade libadc).

Verifie qu'on pilote les solveurs concrets depuis Python sans rien savoir des
templates C++ : construction par config, pas de temps, invariants physiques
(masse conservee, quantite de mouvement nulle pour la gravite), et champ rendu
en tableau numpy de la bonne forme. PYTHONPATH pointe sur le dossier du .so.
"""
import sys
import numpy as np
import adc

fails = 0


def chk(cond, what):
    global fails
    if not cond:
        print("FAIL", what)
        fails += 1


# --- DiocotronSolver ---
cfg = adc.DiocotronConfig()
cfg.n = 64
ds = adc.DiocotronSolver(cfg)
m0 = ds.mass()
for _ in range(5):
    ds.step(0.01)
rho = ds.density()
print(f"DiocotronSolver : shape={rho.shape} masse {m0:.6e} -> {ds.mass():.6e}")
chk(isinstance(rho, np.ndarray) and rho.shape == (64, 64), "diocotron_density_numpy")
chk(abs(ds.mass() - m0) < 1e-9, "diocotron_masse_conservee")

# --- EulerPoissonSolver, backend FFT ---
ec = adc.EulerPoissonConfig()
ec.n = 64
ec.use_fft = True
es = adc.EulerPoissonSolver(ec)
em0 = es.mass()
for _ in range(5):
    es.step(0.004)
print(f"EulerPoissonSolver(FFT) : masse={es.mass():.6e} "
      f"p=({es.total_momentum(0):.2e}, {es.total_momentum(1):.2e})")
chk(abs(es.mass() - em0) < 1e-9, "ep_masse_conservee")
chk(abs(es.total_momentum(0)) < 1e-9, "ep_qte_mouvement_nulle")
chk(es.density().shape == (64, 64), "ep_density_numpy")

# --- EulerPoissonSolver, couplage plasma repulsif (InteractionKind) ---
chk(hasattr(adc, "InteractionKind"), "interaction_kind_expose")
pc = adc.EulerPoissonConfig()
pc.n = 64
pc.interaction = adc.InteractionKind.Plasma  # repulsif : Langmuir + Coulomb
ps = adc.EulerPoissonSolver(pc)
pm0 = ps.mass()
for _ in range(20):
    ps.step(0.004)
print(f"EulerPoissonSolver(Plasma) : masse={ps.mass():.6e} "
      f"p=({ps.total_momentum(0):.2e}, {ps.total_momentum(1):.2e})")
chk(abs(ps.mass() - pm0) < 1e-9, "ep_plasma_masse_conservee")
chk(abs(ps.total_momentum(0)) < 1e-9, "ep_plasma_qte_mouvement_nulle")

# --- DiocotronSolver, CI bande + pas auto CFL ---
bc = adc.DiocotronConfig()
bc.n = 48
bc.ic = adc.DiocotronIC.Band
db = adc.DiocotronSolver(bc)
bm0 = db.mass()
for _ in range(10):
    db.step_cfl(0.4)  # pas stable choisi par la facade (derive E x B)
print(f"DiocotronSolver(Band) : v_derive={db.max_drift_speed():.3e} "
      f"phi.shape={db.potential().shape} dmasse={abs(db.mass() - bm0):.2e}")
chk(db.potential().shape == (48, 48), "diocotron_potential_numpy")
chk(abs(db.mass() - bm0) < 1e-9, "diocotron_band_masse_conservee")

# --- TwoFluidAPSolver, regime raide (AP) ---
tc = adc.TwoFluidAPConfig()
tc.n = 64
tc.omega_pe = 1e3
tc.omega_pi = 20.0
ts = adc.TwoFluidAPSolver(tc)
tm0 = ts.mass_e()
ts.advance(5.0 / 1e3, 200)  # dt*omega_pe = 5 : explicite exploserait
print(f"TwoFluidAPSolver(raide) : max|dne|={ts.max_dev():.3e} "
      f"max|charge|={ts.max_charge():.3e} dmasse_e={abs(ts.mass_e() - tm0):.2e}")
chk(ts.density_e().shape == (64, 64), "tfap_density_numpy")
chk(ts.max_dev() < 0.1, "tfap_AP_borne")
chk(ts.max_charge() < 0.1, "tfap_quasi_neutre")
chk(abs(ts.mass_e() - tm0) < 1e-7, "tfap_masse_conservee")

# --- TwoFluidAPSolver, continuite upwind (flux de masse Rusanov anti-Gibbs) ---
uc = adc.TwoFluidAPConfig()
uc.n = 64
uc.omega_pe = 1e3
uc.omega_pi = 20.0
uc.upwind_continuity = True
us = adc.TwoFluidAPSolver(uc)
um0 = us.mass_e()
us.advance(5.0 / 1e3, 200)
print(f"TwoFluidAPSolver(upwind) : max|dne|={us.max_dev():.3e} dmasse_e={abs(us.mass_e() - um0):.2e}")
chk(us.max_dev() < 0.1, "tfap_upwind_borne")
chk(abs(us.mass_e() - um0) < 1e-7, "tfap_upwind_masse_conservee")

# --- DiocotronAmrSolver, AMR multi-patch + regrid Berger-Rigoutsos ---
ac = adc.DiocotronAmrConfig()
ac.n = 64
ac.regrid_every = 10
asim = adc.DiocotronAmrSolver(ac)
am0 = asim.mass()
for _ in range(30):
    asim.step_cfl(0.4)
arho = asim.density()
print(f"DiocotronAmrSolver : patches={asim.n_patches()} shape={arho.shape} "
      f"dmasse={abs(asim.mass() - am0):.2e}")
chk(arho.shape == (64, 64), "diocotron_amr_density_numpy")
chk(np.isfinite(arho).all(), "diocotron_amr_density_finite")
chk(asim.n_patches() >= 1, "diocotron_amr_a_des_patchs")
chk(abs(asim.mass() - am0) < 1e-9, "diocotron_amr_masse_conservee")

# --- TwoFluidAPSolver, magnetise (rotation cyclotron, B hors-plan) ---
mc = adc.TwoFluidAPConfig()
mc.n = 64
mc.omega_ce = 4.0
mc.omega_ci = 0.2
ms = adc.TwoFluidAPSolver(mc)
mm0 = ms.mass_e()
ms.advance(0.01, 100)
print(f"TwoFluidAPSolver(magnetise) : max|dne|={ms.max_dev():.3e} dmasse_e={abs(ms.mass_e() - mm0):.2e}")
chk(abs(ms.mass_e() - mm0) < 1e-7, "tfap_magnetise_masse_conservee")

# --- MultiSpeciesSolver : composition deux fluides (electrons Euler + ions
# isothermes + Poisson) pilotee depuis Python (CoupledSystem + SystemCoupler en C++) ---
msc = adc.MultiSpeciesConfig()
msc.n = 32
msc.eps = 0.02
mss = adc.MultiSpeciesSolver(msc)
mse0, msi0 = mss.mass_e(), mss.mass_i()
chk(mss.max_charge() > 0.0, "multispecies_charge_nonzero")
mss.advance(0.001, 8)
ne, phi = mss.density_e(), mss.potential()
print(f"MultiSpeciesSolver : ne.shape={ne.shape} max|charge|={mss.max_charge():.3e} "
      f"dmasse_e={abs(mss.mass_e() - mse0):.2e} dmasse_i={abs(mss.mass_i() - msi0):.2e}")
chk(ne.shape == (32, 32) and phi.shape == (32, 32), "multispecies_arrays_numpy")
chk(np.isfinite(ne).all() and np.isfinite(phi).all(), "multispecies_finite")
chk(abs(mss.mass_e() - mse0) < 1e-9, "multispecies_masse_e_conservee")
chk(abs(mss.mass_i() - msi0) < 1e-9, "multispecies_masse_i_conservee")

# --- Simulation : composition a l'EXECUTION (on ajoute les especes une a une) ---
sc = adc.SimulationConfig()
sc.n = 32
sim = adc.Simulation(sc)
sim.add_species("electrons", "diocotron", -1.0)
sim.add_species("ions", "diocotron", 1.0)
chk(sim.n_species() == 2, "simulation_two_species")
xs = (np.arange(32) + 0.5) / 32.0
ne = 1.0 + 0.1 * np.cos(2 * np.pi * xs)[None, :] * np.ones((32, 1))
sim.set_density("electrons", ne)
sim.set_density("ions", np.ones((32, 32)))
sim.solve_fields()
phi = sim.potential()
se0, si0 = sim.mass("electrons"), sim.mass("ions")
sim.advance(0.002, 10)
print(f"Simulation : n_species={sim.n_species()} phimax={np.abs(phi).max():.3e} "
      f"dmasse_e={abs(sim.mass('electrons') - se0):.2e} dmasse_i={abs(sim.mass('ions') - si0):.2e}")
chk(np.abs(phi).max() > 1e-6, "simulation_potential_nonzero")
chk(sim.density("electrons").shape == (32, 32), "simulation_density_numpy")
chk(abs(sim.mass("electrons") - se0) < 1e-10, "simulation_masse_e_conservee")
chk(abs(sim.mass("ions") - si0) < 1e-10, "simulation_masse_i_conservee")

# Simulation HETEROGENE : electrons Euler (4 var) + ions isothermes (3 var) au runtime.
hc = adc.SimulationConfig()
hc.n = 32
hs = adc.Simulation(hc)
hs.add_species("electrons", "electron_euler", -1.0)
hs.add_species("ions", "ion_isothermal", 1.0)
hs.set_density("electrons", 1.0 + 0.01 * np.cos(2 * np.pi * xs)[None, :] * np.ones((32, 1)))
hs.set_density("ions", np.ones((32, 32)))
he0, hi0 = hs.mass("electrons"), hs.mass("ions")
hs.advance(0.001, 6)
print(f"Simulation(hetero) : 4var+3var dmasse_e={abs(hs.mass('electrons') - he0):.2e} "
      f"dmasse_i={abs(hs.mass('ions') - hi0):.2e}")
chk(abs(hs.mass("electrons") - he0) < 1e-10, "simulation_hetero_masse_e")
chk(abs(hs.mass("ions") - hi0) < 1e-10, "simulation_hetero_masse_i")

if fails == 0:
    print("OK test_bindings")
sys.exit(0 if fails == 0 else 1)
