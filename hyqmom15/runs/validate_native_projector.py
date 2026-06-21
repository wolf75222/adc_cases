#!/usr/bin/env python3
"""Native relaxation15 projector validation: compiled m.projection vs oracle.

ADC-275. The native projector (model.py::build_projection, emitted via
m.projection / ADC-177) is the
production path; relaxation.py stays the reference oracle. This script compiles the generated
projection brick (project(U, aux), via emit_cpp_brick + a tiny C++ main) and checks the acceptance
criteria of the issue against the relax goldens and the Python oracle. It needs only a C++ compiler
and the adc headers (no compiled _adc, no Kokkos) -- the pure-Python dsl emits the brick.

Sections:
  (A) cell  : compiled project(U) == relax15 on the 12 relax goldens (branches 0-4), tol 1e-10.
  (B) field : compiled projection over a (15, ny, nx) field == relax_field (Python oracle).
  (C) Ma=20 : the native non-realizable cell rate drops as with relax_field.
  (D) no-reg: projection=False emits no project hook (bit-identical transport).

Run:
    ADC_INCLUDE=<adc_cpp>/include python3 hyqmom15/runs/validate_native_projector.py
(ADC_INCLUDE defaults to ../adc_cpp/include relative to the repo if unset.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # hyqmom15/ : model, relaxation, gen_states

import model as MOD  # noqa: E402
import relaxation as R  # noqa: E402

INCLUDE = os.environ.get(
    "ADC_INCLUDE",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "adc_cpp", "include")),
)

fails = 0


def chk(cond: bool, label: str, detail: str = "") -> None:
    global fails
    print(
        "  [%s] %s%s"
        % ("OK " if cond else "XX ", label, ("  " + detail) if detail else "")
    )
    if not cond:
        fails += 1


def _p2p2_lammin(m4: np.ndarray) -> float:
    """Smallest eigenvalue of p2p2 for a moment vector (realizability gauge)."""
    _, s = R.m2cs4(m4)
    p2 = R.p2p2_2d(
        s[(0, 3)],
        s[(0, 4)],
        s[(1, 1)],
        s[(1, 2)],
        s[(1, 3)],
        s[(2, 1)],
        s[(2, 2)],
        s[(3, 0)],
        s[(3, 1)],
        s[(4, 0)],
    )
    return float(np.sort(np.real(np.linalg.eigvals(p2)))[0])


def _compile_projector(ma: float, lamin: float, tmp: str, cxx: str) -> str:
    """Emit the projection brick for ma and compile a tiny main per cell.

    The generated main calls project(U, a) for each cell passed on argv.
    """
    m = MOD.build_moment_model(
        "brick_ma%d" % int(ma),
        robust=False,
        exact_speeds=True,
        projection=True,
        Ma=ma,
        lamin=lamin,
    )
    hpp = os.path.join(tmp, "brick_ma%d.hpp" % int(ma))
    with open(hpp, "w") as f:
        f.write(m._m.emit_cpp_brick(name="ProjMa%d" % int(ma)))
    main = os.path.join(tmp, "main_ma%d.cpp" % int(ma))
    with open(main, "w") as f:
        f.write(
            "#include <cstdio>\n#include <cstdlib>\n"
            "#include <adc/core/types.hpp>\n#include <adc/core/state.hpp>\n"
            "#include <adc/core/variables.hpp>\n"
            '#include "brick_ma%d.hpp"\n' % int(ma)
            + "int main(int argc, char** argv){\n"
            "  adc_generated::ProjMa%d m; adc::Aux a{};\n" % int(ma)
            + '  std::FILE* fp=std::fopen(argv[1],"w");\n'
            "  for(int c=2;c<argc;c+=15){\n"
            "    adc::StateVec<15> U; for(int k=0;k<15;k++) U[k]=atof(argv[c+k]);\n"
            "    auto P=m.project(U,a);\n"
            '    for(int k=0;k<15;k++) std::fprintf(fp,"%.17g ",(double)P[k]);\n'
            '    std::fprintf(fp,"\\n");\n  }\n  std::fclose(fp); return 0;\n}\n'
        )
    exe = os.path.join(tmp, "main_ma%d" % int(ma))
    cp = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-I", INCLUDE, main, "-o", exe],
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        print(cp.stderr[:3000])
        raise SystemExit("compile failed Ma=%d" % int(ma))
    return exe


def _run(exe: str, field2d: np.ndarray, tmp: str) -> np.ndarray:
    """field2d: (15, N) -> (15, N) projected by the compiled brick."""
    n = field2d.shape[1]
    out = os.path.join(tmp, "out.txt")
    args = [exe, out]
    for c in range(n):
        args += ["%.17g" % field2d[k, c] for k in range(15)]
    subprocess.run(args, check=True)
    arr = np.loadtxt(out)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.T


def main() -> int:
    global fails
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip: no C++ compiler or adc headers (set ADC_INCLUDE)")
        return 0
    inm = np.loadtxt(
        os.path.join(os.path.dirname(HERE), "golden", "golden_relax_in.csv"), delimiter=","
    )
    meta = np.loadtxt(
        os.path.join(os.path.dirname(HERE), "golden", "golden_relax_meta.csv"), delimiter=","
    )
    lamin0 = float(meta[0, 0])
    fn = R.make_corner_eigs()
    tmp = tempfile.mkdtemp()
    try:
        exe = {
            ma: _compile_projector(ma, lamin0, tmp, cxx) for ma in (2.0, 20.0)
        }
        print("compiled both projectors")

        print("== (A) compiled project == relax15 on 12 goldens (tol 1e-10) ==")
        worst_a = 0.0
        for ma in (2.0, 20.0):
            idx = [t for t in range(inm.shape[0]) if float(meta[t, 1]) == ma]
            got = _run(exe[ma], inm[idx].T.copy(), tmp)
            for k, t in enumerate(idx):
                ref = R.relax15(inm[t], lamin0, ma, corner_eigs=fn)
                d = float(np.max(np.abs(got[:, k] - ref)))
                worst_a = max(worst_a, d)
                print(
                    "  state %2d br%d Ma=%2d  absdiff %.3e"
                    % (t, int(meta[t, 2]), int(ma), d)
                )
        chk(
            worst_a < 1e-10,
            "compiled project == relax15 (worst %.3e)" % worst_a,
        )

        print(
            "== (B) compiled field projection == relax_field on (15,ny,nx) =="
        )
        ny, nx = 4, 5
        field = np.empty((15, ny, nx))
        for j in range(ny):
            for i in range(nx):
                field[:, j, i] = inm[(j * nx + i) % inm.shape[0]]
        got_b = _run(exe[2.0], field.reshape(15, -1), tmp).reshape(15, ny, nx)
        ref_b = R.relax_field(field, lamin0, 2.0, corner_eigs=fn)
        d_b = float(np.max(np.abs(got_b - ref_b)))
        chk(d_b < 1e-10, "compiled field == relax_field (worst %.3e)" % d_b)

        print("== (C) Ma=20 non-realizable rate drops like relax_field ==")
        idx20 = [t for t in range(inm.shape[0]) if float(meta[t, 1]) == 20.0]
        stress = inm[idx20].T.copy()

        def rate(a):
            return (
                sum(
                    1
                    for k in range(a.shape[1])
                    if _p2p2_lammin(a[:, k]) <= lamin0
                )
                / a.shape[1]
            )

        before = rate(stress)
        nat = rate(_run(exe[20.0], stress, tmp))
        orc = rate(
            R.relax_field(
                stress.reshape(15, 1, -1), lamin0, 20.0, corner_eigs=fn
            ).reshape(15, -1)
        )
        print(
            "  non-realizable rate before=%.3f native=%.3f oracle=%.3f"
            % (before, nat, orc)
        )
        chk(
            abs(nat - orc) < 1e-9 and nat <= before + 1e-12,
            "native rate == oracle and not increased",
        )

        print(
            "== (D) no-regression : projection=False emits no project hook =="
        )
        m_np = MOD.build_moment_model(
            "noproj", robust=False, exact_speeds=True, projection=False
        )
        src_np = m_np._m.emit_cpp_brick(name="NoProj")
        m_p = MOD.build_moment_model(
            "withproj",
            robust=False,
            exact_speeds=True,
            projection=True,
            Ma=2.0,
            lamin=lamin0,
        )
        src_p = m_p._m.emit_cpp_brick(name="WithProj")
        chk(
            "State project" not in src_np, "projection=False -> no project hook"
        )
        chk("State project" in src_p, "projection=True -> project hook present")
        chk(
            "dense_eig.hpp" in src_p,
            "projector brick includes dense_eig.hpp (eig witness)",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("FAILS = %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
