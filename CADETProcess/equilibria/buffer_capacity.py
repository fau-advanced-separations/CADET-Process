from collections import defaultdict
import copy

import numpy as np
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import MassActionLaw


# def check_charges(charges):
#     c_min = min(charges)
#     c_max = max(charges)
#     c_comparison = set(range(c_min, c_max+1))

#     if len(c_comparison) != len(charges) or set(charges) != c_comparison:
#         raise CADETProcessError("Charges are not valid")

# def order_by_charge(charges, k_eq, buffer):
#     indices = np.argsort(charges)

#     c, k, b = zip(*sorted(zip(charges, k_eq, buffer)))

#     return c, k, b


# def nu(c_acid_M, c_H_M):
#     n = len(c_acid_M)-1
#     nu = c_acid_M[0]/c_H_M**n

#     return nu


def c_species_nu(pKa, pH):
    """Compute normalized acid species concentration at given pH.

    Parameters
    ----------
    pKa : list
        List of pKa values
    pH : float or list of floats.
        pH value

    Returns
    -------
    c_species_nu : np.array
        Normalized acid species concentration.
    """
    pKa = np.array([1.0] + pKa)
    k_eq = 10**(-pKa)
    n = len(k_eq)

    c_H = 10**(-pH)
    c_species_nu = np.zeros((n, len(pH)))

    for j in range(n):
        k = np.prod(k_eq[0:j+1])
        c_species_nu[j] = k*c_H**(n-j)

    return c_species_nu

def c_total_nu(pKa, pH):
    """Compute normalized total acid concentration at given pH.

    Parameters
    ----------
    pKa : list
        List of pKa values
    pH : float or list of floats.
        pH value

    Returns
    -------
    c_total_nu : np.array
        Normalized acid species concentration.
    """
    return sum(c_species_nu(pKa, pH))

def z_total_nu(pKa, pH):
    """Compute normalized total charge at given pH.

    Parameters
    ----------
    pKa : list
        List of pKa values
    pH : float or list of floats.
        pH value

    Returns
    -------
    z_total_nu : np.array
        Normalized acid species concentration.
    """
    c = c_species_nu(pKa, pH)

    return np.dot(np.arange(len(c)), c)

def eta(pKa, pH):
    """Compute degree of dissociation at given pH.

    Parameters
    ----------
    pKa : list
        List of pKa values
    pH : float or list of floats.
        pH value

    Returns
    -------
    eta : np.array
        Degree of dissociation.
    """
    return z_total_nu(pKa, pH)/c_total_nu(pKa, pH)

def alpha(pKa, pH):
    """Compute degree of protolysis at given pH.

    Parameters
    ----------
    pKa : list
        List of pKa values
    pH : float or list of floats.
        pH value

    Returns
    -------
    alpha : np.array
        Degree of protolysis.
    """
    return c_species_nu(pKa, pH)/c_total_nu(pKa, pH)

def beta(c_acid, pKa, pH):
    """Compute buffer capacity of acid at given pH.

    Parameters
    ----------
    c_acid : TYPE
        DESCRIPTION.
    pKa : list
        List of pKa values
    pH : float or list of floats.
        pH value

    Returns
    -------
    beta : np.array
        Buffer capacity.
    """
    a = alpha(pKa, pH)
    beta = np.zeros(len(pH),)

    n = len(c_acid)
    for j in range(1, n):
        for i in range(0, j):
            print(f"j:{j}, i:{i}")
            beta += (j-i)**2 * a[j] * a[i]

    beta *= np.log(10) * sum(c_acid)

    return beta

def beta_water(pH):
    """Compute buffer capacity of water.

    Parameters
    ----------
    pH : float or list of floats.
        pH value

    Returns
    -------
    beta_water
        Buffer capacity of water.
    """
    c_H = 10**(-pH)
    return np.log(10)*(10**(-14)/c_H + c_H)

def buffer_capacity(
        reaction_system,
        buffer, pH,
        components=None,
        ):
    """Calculate buffer capacity at given buffer concentration and pH.
    
    Parameters
    ----------
    reaction_system : ReactionModel
        Reaction system with deprotonation reactions.
    buffer : list
        Acid concentrations in mM.
    pH : float or array
        pH value of buffer.
    components : list, optional
        List of components to be considered in buffer capacity calculation.
        If None, all components are considerd.

    Returns
    -------
    buffer_capacity : np.array
        Buffer capacity in mM for individual acid components.
        To get overall buffer capacity, component capacities must be summed up.
    """
    buffer_M = np.array([c*1e-3 for c in buffer])

    pH = np.asarray(pH)
    scalar_input = False
    if pH.ndim == 0:
        pH = pH[None]  # Makes x 1D
        scalar_input = True

    component_system = copy.deepcopy(reaction_system.component_system)

    indices = component_system.indices
    if components is not None:
        for comp in indices.copy():
            if comp not in components:
               indices.pop(comp) 

    try:
        proton_index = indices.pop('H+')
    except ValueError:
        raise CADETProcessError("Could not find proton in component system")
        
    pKa = defaultdict(list)
    for r in reaction_system.reactions:
        reaction_indices = np.where(r.stoich)[0]
        for comp, i in indices.items():
            if not all(r_i in i + proton_index for r_i in reaction_indices):
                continue

            pKa[comp].append(-np.log10(r.k_eq*1e-3))
        
    c_acids_M = {
        comp: buffer_M[i].tolist()
        for comp, i in indices.items()
    }
    
    buffer_capacity = np.zeros((len(pH), len(c_acids_M)+1))
    buffer_capacity[:,0] = beta_water(pH)

    for i, comp in enumerate(indices):
        buffer_capacity[:,i+1] = beta(c_acids_M[comp], pKa[comp], pH)
        
    buffer_capacity *= 1e3
    if scalar_input:
        return np.squeeze(buffer_capacity)
    
    return buffer_capacity

def ionic_strength(component_system, buffer):
    """Compute ionic strength.

    Parameters
    ----------
    buffer : list
        Buffer concentrations in mM.
    component_system : ComponentSystem
        Component system; must contain charges.

    Returns
    -------
    i: np.array
        Ionic strength of buffer

    """
    if not isinstance(component_system, ComponentSystem):
        raise TypeError("Expected ComponentSystem")
    if len(buffer) != component_system.n_comp:
        raise CADETProcessError("Number of components does not match")

    buffer = np.asarray(buffer)
    z = np.asarray(component_system.charges)
    return 1/2 * np.sum(buffer*z**2)

def plot_buffer_capacity(pH, buffer_capacity, title=None):
    plt.figure()

    for i in range(buffer_capacity.shape[0]):
        plt.plot(pH, buffer_capacity[i, :])

    plt.plot(pH, np.sum(buffer_capacity, 0), 'k*')

    plt.xlabel('pH')
    plt.ylabel('Buffer capacity / mM')

    if title is not None:
        plt.title(title)


if __name__ == '__main__':
    foo = ionic_strength(np.array([0.1, 0.05, 0.02, 0.02]),
                   np.array([1, -2, 1, -1]))
    print(foo)

