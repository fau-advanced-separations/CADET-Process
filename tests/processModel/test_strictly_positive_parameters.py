"""Parameters that CADET-Core requires to be strictly positive reject zero."""

import pytest
from CADETProcess.processModel import (
    BiStericMassAction,
    ComponentSystem,
    GeneralizedIonExchange,
    GeneralRateModel,
    LumpedRateModelWithoutPores,
    LumpedRateModelWithPores,
    MultistateStericMassAction,
    SelfAssociation,
    SimplifiedMultistateStericMassAction,
    StericMassAction,
)
from CADETProcess.simulator.cadetAdapter import SolverTimeIntegratorParameters

SMA_TYPE_MODELS = [
    StericMassAction,
    SelfAssociation,
    BiStericMassAction,
    MultistateStericMassAction,
    SimplifiedMultistateStericMassAction,
    GeneralizedIonExchange,
]

CASES = [
    *(
        (model, parameter)
        for model in SMA_TYPE_MODELS
        for parameter in (
            "reference_liquid_phase_conc",
            "reference_solid_phase_conc",
        )
    ),
    (SimplifiedMultistateStericMassAction, "characteristic_charge_salt"),
    (LumpedRateModelWithoutPores, "total_porosity"),
    *(
        (model, parameter)
        for model in (LumpedRateModelWithPores, GeneralRateModel)
        for parameter in (
            "bed_porosity",
            "particle_porosity",
            "particle_radius",
            "pore_accessibility",
        )
    ),
]


@pytest.mark.parametrize(
    ("model", "parameter"),
    CASES,
    ids=[f"{model.__name__}.{parameter}" for model, parameter in CASES],
)
def test_zero_is_rejected(model, parameter):
    instance = model(ComponentSystem(2), name="test")

    with pytest.raises(ValueError):
        setattr(instance, parameter, 0)


@pytest.mark.parametrize(
    ("model", "parameter"),
    CASES,
    ids=[f"{model.__name__}.{parameter}" for model, parameter in CASES],
)
def test_small_positive_value_is_accepted(model, parameter):
    instance = model(ComponentSystem(2), name="test")

    setattr(instance, parameter, 1e-6)


def test_single_zero_entry_is_rejected():
    column = GeneralRateModel(ComponentSystem(2), name="test")

    with pytest.raises(ValueError):
        column.pore_accessibility = [1, 0]


def test_absolute_tolerance_zero_is_rejected():
    parameters = SolverTimeIntegratorParameters()

    with pytest.raises(ValueError):
        parameters.abstol = 0
