import pytest

from tests.integration.create_LWE import create_lwe


@pytest.fixture
def lwe_process():
    return create_lwe()


@pytest.fixture(params=[
    "GeneralRateModel",
    "LumpedRateModelWithoutPores",
    "LumpedRateModelWithPores",
    "Cstr",
])
def lwe_process_all_units(request):
    return create_lwe(unit_type=request.param)
