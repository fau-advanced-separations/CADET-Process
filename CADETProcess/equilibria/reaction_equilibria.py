from typing import Optional, Sequence

import numpy as np

from CADETProcess.processModel import MassActionLaw

from . import ptc


def calculate_buffer_equilibrium(
    buffer: Sequence[float],
    reaction_system: MassActionLaw,
    constant_indices: Optional[Sequence[int]] = None,
    reinit: bool = True,
    verbose: bool = False,
) -> list[float]:
    """
    Calculate buffer equilibrium for given concentration.

    Parameters
    ----------
    buffer : list of floats
        Buffer concentration in mM
    reaction_system : MassActionLaw
        Reaction rates and stoichiometric matrix for calculating equilibrium.
    constant_indices : list, optional
        Indices of fixed target concentration (e.g. proton concentration/pH).
    reinit: Bool, optional
        If True, run CADET with initial values to get 'smooth' initial values
    verbose : Bool, optional
        If True, print information at every ptc iteration.

    Returns
    -------
    sol : list of floats.
        Buffer equilbrium concentrations
    """

    def residual(c: np.ndarray) -> np.ndarray:
        return dydx_mal(
            c,
            reaction_system,
            constant_indices,
            buffer.copy(),
        )

    def jacobian(c: np.ndarray) -> np.ndarray:
        return jac_mal(
            c,
            reaction_system,
            constant_indices,
            buffer.copy(),
        )

    stats, sol, res, k = ptc(
        np.array(buffer.copy()),
        residual,
        jacobian,
        1e-4,   # init step size
        1e-14,  # tolerance (scaled l2)
        quiet=not (verbose),
        maxIter=10000,
    )

    return np.round(np.abs(sol), 14).tolist()


def dydx_mal(
    c: np.ndarray,
    reaction_system: MassActionLaw,
    constant_indices: Optional[Sequence[int]] = None,
    c_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the time derivative of concentrations in a mass action law system.

    Parameters
    ----------
    c : np.ndarray
        Concentration vector of components.
    reaction_system : MassActionLaw
        Reaction rates and stoichiometric matrix for calculating equilibrium.
    constant_indices : Optional[Sequence[int]]
        Indices of components whose concentrations are to be held constant.
    c_init : Optional[np.ndarray]
        Initial concentration vector (used to fix constants if constant_indices is provided).

    Returns
    -------
    dydx : np.ndarray
        Time derivative of concentration vector.
    """
    cc = np.asarray(c, dtype="float64")
    if constant_indices is not None:
        if c_init is None:
            c_init = c
        for comp in constant_indices:
            cc[comp] = c_init[comp]

    exp_fwd = reaction_system.exponents_fwd
    exp_bwd = reaction_system.exponents_bwd
    k_fwd = reaction_system.k_fwd
    k_bwd = reaction_system.k_bwd

    r = np.zeros(reaction_system.n_reactions)

    for r_i in range(reaction_system.n_reactions):
        fwd_indices = np.where(exp_fwd[:, r_i] > 0.0)
        prod = np.prod(cc[fwd_indices] ** exp_fwd[:, r_i][fwd_indices])
        r_fwd = k_fwd[r_i] * prod

        bwd_indices = np.where(exp_bwd[:, r_i] > 0.0)
        prod = np.prod(cc[bwd_indices] ** exp_bwd[:, r_i][bwd_indices])
        r_bwd = k_bwd[r_i] * prod

        r[r_i] = r_fwd - r_bwd

    dydx = np.dot(reaction_system.stoich, r)

    if constant_indices is not None:
        for comp in constant_indices:
            dydx[comp] = 0

    return dydx


def jac_mal(
    c: np.ndarray,
    reaction_system: MassActionLaw,
    constant_indices: Optional[Sequence[int]] = None,
    c_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the Jacobian of a mass action law reaction system at given concentrations.

    Parameters
    ----------
    c : np.ndarray
        Current concentrations of components.
    reaction_system : MassActionLaw
        Reaction system object containing stoichiometry, exponents, and rate constants.
    constant_indices : Optional[Sequence[int]]
        Indices of components to be treated as constant
        (i.e., their derivatives are zeroed out).
    c_init : Optional[np.ndarray]
        Initial concentration vector to reset constants if `constant_indices` is provided.

    Returns
    -------
    jac : np.ndarray
        Jacobian matrix (n_comp x n_comp) of the rate equations.

    """
    cc = np.asarray(c, dtype="float64")

    if constant_indices is not None:
        if c_init is None:
            c_init = c
        for comp in constant_indices:
            cc[comp] = c_init[comp]

    exp_fwd = reaction_system.exponents_fwd
    exp_bwd = reaction_system.exponents_bwd
    k_fwd = reaction_system.k_fwd
    k_bwd = reaction_system.k_bwd

    jac_r = np.zeros((reaction_system.n_reactions, reaction_system.n_comp))

    for r_i in range(reaction_system.n_reactions):
        fwd_indices = np.where(exp_fwd[:, r_i] > 0.0)
        j_fwd = np.zeros_like(cc)
        j_fwd[fwd_indices] = k_fwd[r_i]

        bwd_indices = np.where(exp_bwd[:, r_i] > 0.0)
        j_bwd = np.zeros_like(cc)
        j_bwd[bwd_indices] = k_bwd[r_i]

        for comp in range(len(c)):
            exponents_fwd = exp_fwd[:, r_i]
            exponents_bwd = exp_bwd[:, r_i]
            if exponents_fwd[comp] > 0.0:
                exp_derivative = exponents_fwd.copy()
                exp_derivative[comp] -= 1.0
                prod = np.prod(cc**exp_derivative)
                j_fwd[comp] *= exponents_fwd[comp] * prod
            if exponents_bwd[comp] > 0.0:
                exp_derivative = exponents_bwd.copy()
                exp_derivative[comp] -= 1.0
                prod = np.prod(cc**exp_derivative)
                j_bwd[comp] *= exponents_bwd[comp] * prod

        jac_r[r_i, :] = j_fwd - j_bwd

    jac = np.dot(reaction_system.stoich, jac_r)

    if constant_indices is not None:
        for comp in constant_indices:
            jac[comp, :] = 0

    return jac
