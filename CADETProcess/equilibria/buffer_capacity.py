from collections import defaultdict
import copy

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess import plotting
from CADETProcess.processModel import ComponentSystem


def preprocessing(reaction_system, buffer, pH=None, components=None):
    buffer = np.array(buffer, ndmin=2)
    buffer_M = 1e-3*buffer

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

    if pH is None:
        pH = -np.log10(buffer_M[:, proton_index]).reshape((-1))
    else:
        pH = np.asarray(pH, dtype='float64')

    scalar_input = False
    if pH.ndim == 0:
        pH = pH[None]  # Makes x 1D
        scalar_input = True

    pKa = defaultdict(list)
    for r in reaction_system.reactions:
        reaction_indices = np.where(r.stoich)[0]
        for comp, i in indices.items():
            if not all(r_i in i + proton_index for r_i in reaction_indices):
                continue

            pKa[comp].append(-np.log10(r.k_eq*1e-3))

    c_acids_M = {
        comp: buffer_M[:, i]
        for comp, i in indices.items()
        if comp in pKa
    }

    for comp in indices.copy():
        if comp not in pKa:
            indices.pop(comp)

    return pKa, c_acids_M, pH, indices, scalar_input


def c_species_nu(pKa, pH):
    """Compute normalized acid species concentration at given pH.

    Parameters
    ----------
    pKa : list
        List of pKa values.
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

    c_H = np.power(10, -pH)
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


def charge_distribution(reaction_system, pH, components=None):
    """Calculate charge distribution at given pH.

    Parameters
    ----------
    reaction_system : ReactionModel
        Reaction system with deprotonation reactions.
    pH : float or array
        pH value of buffer.
    components : list, optional
        List of components to be considered in buffer capacity calculation.
        If None, all components are considered.

    Returns
    -------
    charge_distribution : np.array
        Degree of protolysis; ratio of the concentration of the species to the
        total concentration.

    """
    buffer = reaction_system.n_comp * [1]
    pKa, c_acids_M, pH, indices, scalar_input = preprocessing(
        reaction_system, buffer, pH, components
    )

    if components is None:
        z_shape = (len(pH), reaction_system.n_comp - 1)
    else:
        n_comp = 0
        for comp in indices.values():
            n_comp += len(comp)
        z_shape = (len(pH), n_comp)

    z = np.zeros(z_shape)

    counter = 0
    for comp, ind in indices.items():
        z_comp = alpha(pKa[comp], pH)
        for j in range(len(ind)):
            z[:, counter] = z_comp[j, :]
            counter += 1

    if scalar_input:
        return np.squeeze(z)

    return z


def cummulative_charge_distribution(reaction_system, pH, components=None):
    """Calculate cummulative charge at given pH.

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
    cummulative_charge_distribution : np.array
        Degree of dissociation;
    """
    buffer = reaction_system.n_comp * [1]
    pKa, c_acids_M, pH, indices, scalar_input = preprocessing(
        reaction_system, buffer, pH, components
    )

    z_cum = np.zeros((len(pH), len(indices)))

    for i, (comp, ind) in enumerate(indices.items()):
        charges = np.array(reaction_system.component_system.charges)[ind]
        max_charge = max(charges)
        z_cum[:, i] = max_charge - eta(pKa[comp], pH)

    if scalar_input:
        return np.squeeze(z_cum)

    return z_cum


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

    n = c_acid.shape[1]
    for j in range(1, n):
        for i in range(0, j):
            beta += (j-i)**2 * a[j] * a[i]

    beta *= np.log(10) * np.sum(c_acid, axis=1)

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


def buffer_capacity(reaction_system, buffer, pH=None, components=None):
    """Calculate buffer capacity at given buffer concentration and pH.

    Parameters
    ----------
    reaction_system : ReactionModel
        Reaction system with deprotonation reactions.
    buffer : list
        Acid concentrations in mM.
    pH : float or array, optional
        pH value of buffer. If None, value is inferred from buffer entry.
    components : list, optional
        List of components to be considered in buffer capacity calculation.
        If None, all components are considerd.

    Returns
    -------
    buffer_capacity : np.array
        Buffer capacity in mM for individual acid components.
        To get overall buffer capacity, component capacities must be summed up.
    """
    pKa, c_acids_M, pH, indices, scalar_input = preprocessing(
        reaction_system, buffer, pH, components
    )

    buffer_capacity = np.zeros((len(pH), len(c_acids_M)+1))

    for i, comp in enumerate(indices):
        buffer_capacity[:, i] = beta(c_acids_M[comp], pKa[comp], pH)

    buffer_capacity[:, -1] = beta_water(pH)

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

    buffer = np.asarray(buffer, dtype='float64')
    z = np.asarray(component_system.charges)
    return 1/2 * np.sum(buffer*z**2)


@plotting.create_and_save_figure
def plot_buffer_capacity(reaction_system, buffer, pH=None, ax=None):
    """Plot buffer capacity of reaction system over pH at given concentration.

    Parameters
    ----------
    reaction_system : MassActionLaw
        Reaction system with stoichiometric coefficients and reaction rates.
    buffer : list
        Buffer concentration in mM.
    pH : np.array, optional
        Range of pH to be plotted.
    ax : Axes
        Axes to plot on.

    Returns
    -------
    ax : Axes
        Axes object with buffer capacity plot.
    """
    if pH is None:
        pH = np.linspace(0, 14, 101)

    b = buffer_capacity(reaction_system, buffer, pH)
    b_total = np.sum(b, axis=1)

    labels = reaction_system.component_system.names
    labels.remove('H+')

    for i in range(reaction_system.component_system.n_components - 1):
        ax.plot(pH, b[:, i], label=labels[i])

    ax.plot(pH, b[:, -1], label='Water')
    ax.plot(pH, b_total, 'k--', label='Total buffer capacity')

    layout = plotting.Layout()
    layout.x_label = '$pH$'
    layout.y_label = 'buffer capacity / mM'
    layout.y_lim = (0, 1.1*np.max(b_total))

    plotting.set_layout(ax, layout)

    return ax


@plotting.create_and_save_figure
def plot_charge_distribution(
        reaction_system, pH=None, plot_cumulative=False, ax=None):
    """Plot charge distribution of components over pH.

    Parameters
    ----------
    reaction_system : MassActionLaw
        Reaction system with stoichiometric coefficients and reaction rates.
    pH : np.array, optional
        Range of pH to be plotted.
    plot_cumulative : Bool
        If True, only plot cumulative charge of each acid.
    ax : Axes
        Axes to plot on.

    Returns
    -------
    ax : Axes
        Axes object with charge distribution plot.
    """
    if pH is None:
        pH = np.linspace(0, 14, 101)

    layout = plotting.Layout()

    if plot_cumulative:
        c = cummulative_charge_distribution(reaction_system, pH)
        layout.y_label = 'degree of dissociation'
    else:
        c = charge_distribution(reaction_system, pH)
        layout.y_label = 'degree of protolysis'

    if plot_cumulative:
        labels = reaction_system.component_system.names
    else:
        labels = reaction_system.component_system.species

    labels.remove('H+')

    for i, l in zip(c.T, labels):
        ax.plot(pH, i, label=l)

    layout.x_label = '$pH$'
    layout.y_lim = (1.1*np.min(c), 1.1*np.max(c))

    plotting.set_layout(ax, layout)

    return ax
