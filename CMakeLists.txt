cmake_minimum_required(VERSION 3.20)

project(adc_cases
  VERSION 0.0.1
  DESCRIPTION "Applications et cas physiques du solveur adc (modeles, facades, exemples)"
  LANGUAGES CXX C)

# adc_cases = la couche APPLICATION : modeles physiques (diocotron, Euler-Poisson,
# two-fluid), facades compilees (libadc), bindings Python, exemples, tutoriels.
# Le COEUR generique (concepts, coupleur, elliptique, AMR, seams) vit dans adc_cpp,
# tire via FetchContent et consomme par la cible adc::adc.

option(ADC_CASES_BUILD_TESTS    "Build application tests" ON)
option(ADC_CASES_BUILD_EXAMPLES "Build example drivers"   ON)
option(ADC_CASES_BUILD_PYTHON   "Build pybind11 bindings" OFF)
# Backends : propages au coeur adc_cpp (le seam est une propriete de la lib).
option(ADC_USE_OPENMP "Enable the OpenMP dispatch backend" OFF)
option(ADC_USE_KOKKOS "Enable the Kokkos dispatch backend" OFF)
option(ADC_USE_MPI    "Enable the MPI backend"             OFF)
option(ADC_USE_HDF5   "Enable the HDF5 DataWriter"         OFF)
option(ADC_USE_EIGEN  "Build host-side Eigen analysis"     ON)

# Emplacement du coeur. Par defaut le sibling local ../adc_cpp (dev). Surchargeable
# par -DADC_CPP_DIR=/chemin ou (a terme) bascule sur GIT_REPOSITORY.
set(ADC_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../adc_cpp"
    CACHE PATH "Chemin du depot coeur adc_cpp")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Coeur adc_cpp via FetchContent (on ne construit QUE la lib, pas ses tests) ---
# On force les options du coeur AVANT de le rendre disponible : memes backends que
# l'application, mais ni tests ni exemples ni python cote coeur.
set(ADC_USE_OPENMP   ${ADC_USE_OPENMP}   CACHE BOOL "" FORCE)
set(ADC_USE_KOKKOS   ${ADC_USE_KOKKOS}   CACHE BOOL "" FORCE)
set(ADC_USE_MPI      ${ADC_USE_MPI}      CACHE BOOL "" FORCE)
set(ADC_USE_HDF5     ${ADC_USE_HDF5}     CACHE BOOL "" FORCE)
set(ADC_USE_EIGEN    ${ADC_USE_EIGEN}    CACHE BOOL "" FORCE)
set(ADC_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(ADC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ADC_BUILD_PYTHON   OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(adc_cpp SOURCE_DIR ${ADC_CPP_DIR})
FetchContent_MakeAvailable(adc_cpp)

# La norme C++ suit celle du coeur (20 sous Kokkos, 23 sinon).
if(ADC_USE_KOKKOS)
  set(ADC_CASES_CXX_STD 20)
else()
  set(ADC_CASES_CXX_STD 23)
endif()

# --- Couche application (header-only) : modeles + facades headers ---
add_library(adc_cases INTERFACE)
add_library(adc_cases::adc_cases ALIAS adc_cases)
target_include_directories(adc_cases INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(adc_cases INTERFACE adc::adc)        # herite du coeur + backends
target_compile_features(adc_cases INTERFACE cxx_std_${ADC_CASES_CXX_STD})

# --- Facade COMPILEE (libadc) : instancie la pile template une seule fois ---
add_library(adc_cases_solver STATIC
  src/diocotron_solver.cpp
  src/diocotron_amr_solver.cpp
  src/euler_poisson_solver.cpp
  src/two_fluid_ap_solver.cpp)
target_link_libraries(adc_cases_solver PUBLIC adc_cases)
set_target_properties(adc_cases_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(adc_cases::solver ALIAS adc_cases_solver)

if(ADC_CASES_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 2.11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
  endif()
  add_subdirectory(python)
endif()

if(ADC_CASES_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ADC_CASES_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
