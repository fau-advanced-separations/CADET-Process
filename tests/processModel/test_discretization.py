import pytest
from CADETProcess.processModel.discretization import ConsistencySolverParameters

# Solver names are the identifiers in CADET-Core v5.1.2, `src/libcadet/nonlin/`.


@pytest.mark.parametrize("solver_name", ["LEVMAR", "ATRN_RES", "ATRN_ERR", "COMPOSITE"])
def test_consistency_solver_accepts_engine_solver_names(solver_name):
    consistency_solver = ConsistencySolverParameters()

    consistency_solver.solver_name = solver_name

    assert consistency_solver.solver_name == solver_name


def test_consistency_solver_rejects_misspelled_solver_name():
    consistency_solver = ConsistencySolverParameters()

    with pytest.raises(ValueError):
        consistency_solver.solver_name = "ARTN_ERR"


def test_composite_consistency_solver_takes_a_list_of_subsolvers():
    consistency_solver = ConsistencySolverParameters()

    consistency_solver.solver_name = "COMPOSITE"
    consistency_solver.subsolvers = ["ATRN_ERR", "LEVMAR"]

    assert consistency_solver.parameters["subsolvers"] == ["ATRN_ERR", "LEVMAR"]


def test_composite_consistency_solver_accepts_repeated_subsolvers():
    consistency_solver = ConsistencySolverParameters()

    # A repeated solver restarts from where the previous run stopped.
    consistency_solver.subsolvers = ["LEVMAR", "LEVMAR"]

    assert consistency_solver.subsolvers == ["LEVMAR", "LEVMAR"]


@pytest.mark.parametrize(
    "subsolvers",
    [
        ["ATRN_ERR", "ARTN_ERR"],  # unknown names make the engine add a default composite
        ["COMPOSITE"],
    ],
)
def test_composite_consistency_solver_rejects_invalid_subsolvers(subsolvers):
    consistency_solver = ConsistencySolverParameters()

    with pytest.raises(ValueError):
        consistency_solver.subsolvers = subsolvers


def test_consistency_solver_defaults_to_levmar():
    consistency_solver = ConsistencySolverParameters()

    assert consistency_solver.solver_name == "LEVMAR"
