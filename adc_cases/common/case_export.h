#pragma once
/* Export PORTABLE des points d'entree C des cas natifs (charges par ctypes via run.py).
 *
 * Pourquoi : cl/clang-cl en /LD n'exportent RIEN d'une DLL par defaut -- `extern "C"` seul ne suffit
 * PAS sous Windows. Sans __declspec(dllexport), ctypes ne trouve pas tfap_create & co (load_symbols
 * leve "symboles attendus absents"). POSIX : la visibilite est deja "default" ; visibility("default")
 * est inoffensif et garde le symbole meme si la TU est compilee -fvisibility=hidden.
 *
 * Usage : prefixer chaque fonction exposee a ctypes, dans le bloc extern "C" :
 *     extern "C" {
 *       ADC_CASE_EXPORT void* tfap_create(...);
 *     }
 *
 * En-tete C-compatible (les cas exposent une ABI C) : pas de C++ ici.
 */
#if defined(_WIN32)
#define ADC_CASE_EXPORT __declspec(dllexport)
#else
#define ADC_CASE_EXPORT __attribute__((visibility("default")))
#endif
