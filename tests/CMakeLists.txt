# Tests applicatifs : modeles physiques, facades, et integration coeur+modele.
# Lient la couche application adc_cases (qui herite du coeur adc::adc).

function(adc_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE adc_cases::adc_cases)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- Modeles + integration coeur/modele (header-only) ---
adc_add_test(test_diocotron_model)
adc_add_test(test_diocotron_stability)
adc_add_test(test_diocotron_invariants)
adc_add_test(test_interfaces)
adc_add_test(test_euler)
adc_add_test(test_euler_poisson)
adc_add_test(test_euler_poisson_plasma)
adc_add_test(test_magnetic_euler_poisson)
adc_add_test(test_gauss_law)
adc_add_test(test_riemann)
adc_add_test(test_face_fluxes)
adc_add_test(test_spatial_operator)
adc_add_test(test_advect)
adc_add_test(test_coupler)
adc_add_test(test_fft_coupler)
adc_add_test(test_muscl_convergence)
adc_add_test(test_weno2d)
# AMR (exerce avec un modele)
adc_add_test(test_advance_amr)
adc_add_test(test_amr_coupler)
adc_add_test(test_amr_coupler_mp)
adc_add_test(test_amr_cluster_step)
adc_add_test(test_amr_multipatch)
adc_add_test(test_amr_multilevel)
adc_add_test(test_amr_multilevel_mf)
adc_add_test(test_amr_multilevel_multipatch)
adc_add_test(test_amr_reflux)
adc_add_test(test_amr_reflux_mf)
# Two-fluid / AP
adc_add_test(test_two_fluid)
adc_add_test(test_two_fluid_spatial)
adc_add_test(test_two_fluid_ap)
adc_add_test(test_two_fluid_ap_spatial)
adc_add_test(test_two_fluid_ap_2d)
adc_add_test(test_two_fluid_ap_2d_mf)
adc_add_test(test_two_fluid_ap_amplitude)
adc_add_test(test_two_fluid_cyclotron)
adc_add_test(test_two_fluid_boris)

# --- Facade COMPILEE (libadc) ---
add_executable(test_solver test_solver.cpp)
target_link_libraries(test_solver PRIVATE adc_cases::solver)
add_test(NAME test_solver COMMAND test_solver)

# --- Outils d'analyse Eigen (cote hote) ---
if(TARGET adc_eigen)
  add_executable(test_diocotron_theory test_diocotron_theory.cpp)
  target_link_libraries(test_diocotron_theory PRIVATE adc_cases::adc_cases adc_eigen)
  add_test(NAME test_diocotron_theory COMMAND test_diocotron_theory)
endif()

# --- Sortie HDF5 ---
if(ADC_USE_HDF5)
  adc_add_test(test_hdf5_io)
endif()

# --- Tests MPI applicatifs (mpirun -np 4) ---
if(ADC_USE_MPI)
  find_package(MPI REQUIRED)
  function(adc_add_mpi_test name nproc)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE adc_cases::adc_cases)
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
  endfunction()
  adc_add_mpi_test(test_mpi_advect 4)
  adc_add_mpi_test(test_mpi_diocotron 4)
  adc_add_mpi_test(test_mpi_diocotron_amr 4)
  adc_add_mpi_test(test_mpi_overlap 4)
  adc_add_mpi_test(test_mpi_decoarse 4)
  adc_add_mpi_test(test_mpi_amr_multipatch 4)
  adc_add_mpi_test(test_mpi_amr_multipatch3 4)
endif()
