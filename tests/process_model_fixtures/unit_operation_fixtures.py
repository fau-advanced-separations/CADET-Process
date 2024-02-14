import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Cstr, LumpedRateModelWithPores

from binding_model_fixtures import create_isotherm


def create_cstr(
        n_comp,
        V=1e-3,
        binding_model=None,
        binding_model_params=None,
        c=None,
        flow_rate_filter=None
        ):
    component_system = ComponentSystem(n_comp)

    if binding_model is not None:
        binding_model = create_isotherm(binding_model, n_comp, binding_model_params)

    unit = Cstr('cstr', component_system)

    unit.V = V
    unit.flow_rate_filter = flow_rate_filter

    if c is None:
        c = np.zeros((n_comp,))
    unit.c = c

    if binding_model is not None:
        unit.porosity
        unit.q
