# Pilotes MINCES sur la facade compilee (adc_cases::solver) : ils ne font que CI (via
# config), diagnostics (lecture des champs) et I/O. Toute la physique est dans libadc.
add_executable(diocotron diocotron.cpp)
target_link_libraries(diocotron PRIVATE adc_cases::solver)

add_executable(diocotron_column diocotron_column.cpp)
target_link_libraries(diocotron_column PRIVATE adc_cases::solver)

# Demos du MOTEUR bas-niveau (AMR, MPI, theorie) : capacites hors perimetre de la
# facade -> ils pilotent directement la pile generique (adc_cases::adc_cases).
add_executable(diocotron_amr diocotron_amr.cpp)
target_link_libraries(diocotron_amr PRIVATE adc_cases::adc_cases)

add_executable(diocotron_amr3 diocotron_amr3.cpp)
target_link_libraries(diocotron_amr3 PRIVATE adc_cases::adc_cases)

add_executable(diocotron_multipatch diocotron_multipatch.cpp)
target_link_libraries(diocotron_multipatch PRIVATE adc_cases::adc_cases)

add_executable(diocotron_column_amr diocotron_column_amr.cpp)
target_link_libraries(diocotron_column_amr PRIVATE adc_cases::adc_cases)

add_executable(diocotron_highorder diocotron_highorder.cpp)
target_link_libraries(diocotron_highorder PRIVATE adc_cases::adc_cases)

add_executable(magnetic_diocotron magnetic_diocotron.cpp)
target_link_libraries(magnetic_diocotron PRIVATE adc_cases::adc_cases)

add_executable(bench_amr bench_amr.cpp)
target_link_libraries(bench_amr PRIVATE adc_cases::adc_cases)

# Garde de NON-REGRESSION (perf/ressources) cablee sur le bench existant, en mode
# "smoke" : petite taille, GARDE DURE deterministe (derive de masse < 1e-6, sinon
# rc!=0). Le DEBIT reste informatif (jamais asserte) => non flaky. L'executable de
# bench reste SEPARE des TU de test : le bit-identique des tests serie/MPI est
# preserve par construction (aucun test existant ni .hpp du coeur touche).
if(ADC_CASES_BUILD_TESTS)
  add_test(NAME bench_amr_smoke COMMAND bench_amr 64 20 smoke)
endif()

add_executable(diocotron_mpi diocotron_mpi.cpp)
target_link_libraries(diocotron_mpi PRIVATE adc_cases::adc_cases)

if(TARGET adc_eigen)
  add_executable(diocotron_theory diocotron_theory.cpp)
  target_link_libraries(diocotron_theory PRIVATE adc_cases::adc_cases adc_eigen)
endif()

# Demos GPU (Kokkos) : integres au build principal quand le backend Kokkos est
# configure (-DADC_USE_KOKKOS=ON). Ils lient `adc` et heritent du backend ; aucun
# drapeau rajoute. Hors Kokkos, on les ignore (ils n'ont de sens que sur GPU).
if(ADC_USE_KOKKOS)
  add_subdirectory(gpu)
endif()
