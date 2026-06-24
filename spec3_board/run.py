"""Demo "spec3_board" : ecrire un modele "comme au tableau" (Spec 3) et montrer
qu'il s'abaisse vers le noyau operator-first (Spec 2).

Capacite demontree
------------------
La Spec 3 ajoute une facade physico-mathematique (``adc.physics.Model`` +
``adc.math``) qui se lit comme des equations, SANS remplacer le noyau
operator-first de la Spec 2 (``adc.model.Module`` + ``adc.time.Program``). La
facade ne fait que CONSTRUIRE les memes objets ; elle n'a ni registre, ni
scheduler, ni codegen propre. Ce cas le prouve par introspection, cote
application (adc_cases), sans rien compiler en C++ :

  1) on ecrit un modele Euler-Poisson-Lorentz au tableau :
         d_t U = -div F(U) + A(E)U,   -Delta phi = alpha (rho - rho_ref),
         E = -grad phi,   C(B) = operateur lineaire local (Lorentz) ;
  2) on verifie qu'il s'abaisse vers un ``adc.model.Module`` typé : un
     ``StateSpace`` U(rho, mx, my), un ``FieldSpace``, et les operateurs
     ``explicit_rate : (U, Fields) -> Rate(U)`` et
     ``implicit_operator : Fields -> LocalLinearOperator(U, U)`` ;
  3) on ecrit un pas de temps au tableau (``T.fields`` / ``T.rhs`` /
     ``T.solve`` / ``T.commit``) ET la meme chose en operator-first
     (``P.solve_fields`` / ``P.rhs`` / ``P.linear_combine`` /
     ``P.solve_local_linear`` / ``P.commit``), et on EXIGE que les deux
     produisent un IR identique (noeuds ET commits).

C'est la garantie anti-duplication de la Spec 3 : une seule semantique
(operator-first), deux ecritures. Tout le calcul reste C++ a l'execution ;
ici on ne touche que la DESCRIPTION (Python), donc aucun ``_adc`` compile
specifique n'est requis au-dela de l'import du module.

Etat : ce cas n'avance pas une simulation (pas de ``sim.step``) ; il valide la
chaine d'abaissement facade -> IR operator-first. La variante qui COMPILE et
EXECUTE un modele board est un suivi (elle requiert un compilateur + Kokkos).
"""
import adc
from adc.math import sqrt, grad, div, laplacian, ddt, unknown
from adc.physics import Model
from adc.time import Program


def build_board_model():
    """Le modele Euler-Poisson-Lorentz, ecrit au tableau."""
    m = Model("euler_poisson_lorentz")
    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U
    u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
    alpha = m.param("alpha", 1.0)
    rho_ref = m.param("rho_ref", 1.0)
    cs2 = m.param("cs2", 1.0)
    p, c = m.scalar("p", cs2 * rho), m.scalar("c", sqrt(cs2))

    flux = m.flux("F(U)", on=U,
                  x=[mx, mx * u + p, mx * v],
                  y=[my, my * u, my * v + p],
                  waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})

    phi = m.field("phi")
    m.solve_field("fields_from_state",
                  equation=(-laplacian(phi) == alpha * (rho - rho_ref)),
                  outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
                  solver="geometric_mg")
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    a_src = m.source("A(E)U", on=U, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])

    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])

    m.rate("explicit_rate", ddt(U) == -div(flux) + a_src)
    m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    m.check()
    return m


def board_step():
    """Un pas implicite ecrit au tableau."""
    T = Program("board_step")
    dt = T.dt
    U_n = T.state("plasma")
    f_n = T.fields("fields_n", from_state=U_n)
    R_n = T.rhs(name="R_n", state=U_n, fields=f_n, flux=True, sources=["A_E_U"])
    U_star = T.solve("U_star",
                     (T.I - dt * T.linear_source("C_B")) @ unknown("U_star")
                     == U_n + dt * R_n)
    T.commit("plasma", U_star)
    return T


def operator_first_step():
    """Le meme pas, ecrit explicitement en operator-first (Spec 2)."""
    P = Program("operator_first_step")
    dt = P.dt
    U_n = P.state("plasma")
    f_n = P.solve_fields("fields_n", U_n)
    R_n = P.rhs(name="R_n", state=U_n, fields=f_n, flux=True, sources=["A_E_U"])
    op = P.I - dt * P.linear_source("C_B")
    rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
    U_star = P.solve_local_linear("U_star", operator=op, rhs=rhs)
    P.commit("plasma", U_star)
    return P


def _ir(P):
    idx = {id(value): k for k, value in enumerate(P._values)}
    nodes = [(value.vtype, value.op, tuple(idx[id(i)] for i in value.inputs),
              repr(sorted(value.attrs.items())), value.block) for value in P._values]
    commits = sorted((block, idx[id(st)]) for block, st in P.commits().items())
    return (nodes, commits)


def main():
    print("adc", adc.__version__)

    # 1-2) le modele board s'abaisse vers le Module operator-first typé.
    m = build_board_model()
    mod = m.module
    state = mod.state_spaces()["U"]
    assert state.components == ("rho", "mx", "my"), state.components
    ops = set(mod.list_operators())
    assert {"explicit_rate", "implicit_operator", "fields_from_state"} <= ops, ops
    assert mod.operator_registry().get("explicit_rate").kind == "local_rate"
    assert mod.operator_registry().get("implicit_operator").kind == "local_linear_operator"
    print(m.dump_module_ir())

    # 3) board IR == operator-first IR : une seule semantique, deux ecritures.
    board, opf = board_step(), operator_first_step()
    assert _ir(board) == _ir(opf), "board IR must equal the operator-first IR"
    print()
    print(board.dump_operator_ir())

    print("\nOK: la facade Spec 3 s'abaisse vers le noyau operator-first (IR identique).")


if __name__ == "__main__":
    main()
