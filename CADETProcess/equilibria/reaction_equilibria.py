import warnings

import numpy as np

from . import ptc

def calculate_buffer_equilibrium(
        buffer,
        reaction_system,
        constant_indices=None,
        reinit=True,
        verbose=False
    ):
    """Calculate buffer equilibrium for given concentration.
    
    Parameters
    ----------
    buffer : list of floats
        buffer concentration in mM
    reaction_system : MassActionLaw
        reaction rates and stoichiometric matrix for calculating equilibrium.
    constant_indices : list, optional
        Indices of fixed target concentration (e.g. proton concentration/pH).
    reinit: Bool, optional
        if True, run CADET with initial values to get 'smooth' initial values
    verbose : Bool, optional
        if True, print information at every ptc iteration.

    Returns
    -------
    sol : list of floats.
        buffer equilbrium concentrations
    """
    residual = lambda c: dydx_mal(
        c,
        reaction_system,
        constant_indices,
        buffer.copy(),
    )
    jacobian = lambda c: jac_mal(
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
        quiet=not(verbose),
        maxIter=10000,
    )
    
    return np.round(np.abs(sol), 14).tolist()


def dydx_mal(c, reaction_system, constant_indices=None, c_init=None):
    cc = np.asarray(c)
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
        fwd_indices = np.where(exp_fwd[:,r_i] > 0.0)
        r_fwd = k_fwd[r_i] * np.prod(cc[fwd_indices]**exp_fwd[:,r_i][fwd_indices])
        
        bwd_indices = np.where(exp_bwd[:,r_i] > 0.0)
        r_bwd = k_bwd[r_i] * np.prod(cc[bwd_indices]**exp_bwd[:,r_i][bwd_indices])
        
        r[r_i] = r_fwd - r_bwd
        
    dydx = np.dot(reaction_system.stoich, r)
    
    if constant_indices is not None:
        for comp in constant_indices:
            dydx[comp] = 0
        
    return dydx


def jac_mal(c, reaction_system, constant_indices=None, c_init=None):
    cc = np.asarray(c)
    
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
        fwd_indices = np.where(exp_fwd[:,r_i] > 0.0)
        j_fwd = np.zeros_like(cc)
        j_fwd[fwd_indices] = k_fwd[r_i]
        
        bwd_indices = np.where(exp_bwd[:,r_i] > 0.0)
        j_bwd = np.zeros_like(cc)
        j_bwd[bwd_indices] = k_bwd[r_i]
        
        for comp in range(len(c)):
            exponents_fwd = exp_fwd[:,r_i]
            exponents_bwd = exp_bwd[:,r_i]          
            if exponents_fwd[comp] > 0.0:
                exp_derivative = exponents_fwd.copy()
                exp_derivative[comp] -= 1.0
                j_fwd[comp] *= exponents_fwd[comp] * np.prod(cc**exp_derivative)
            if exponents_bwd[comp] > 0.0:
                exp_derivative = exponents_bwd.copy()
                exp_derivative[comp] -= 1.0
                j_bwd[comp] *= exponents_bwd[comp] * np.prod(cc**exp_derivative)
        
        jac_r[r_i, :] = j_fwd - j_bwd
    
    jac = np.dot(reaction_system.stoich, jac_r)
    
    if constant_indices is not None:
        for comp in constant_indices:
            jac[comp,:] = 0
            
    return jac
