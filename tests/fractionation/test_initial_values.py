import pytest
from CADETProcess.fractionation import Fractionator
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel.process import ProcessMeta

from tests.chromatogram_factory import rectangle_chromatogram

# Time axis 0 to 10 with 1001 samples; the window (5, 5.005) covers only the
# sample at t = 5, which initial_values must ignore as a single-sample region.
ISOLATED_SAMPLE = (5, 5.005)


def _collection_windows(windows: list) -> list[tuple[float, float]]:
    """Run initial_values on a one-component rectangle chromatogram."""
    component_system = ComponentSystem(1)
    chromatogram = rectangle_chromatogram(component_system, windows)
    process_meta = ProcessMeta(
        cycle_time=chromatogram.cycle_time,
        V_eluent=1.0,
        V_solid=1.0,
        m_feed=[1.0],
    )
    fractionator = Fractionator([chromatogram], process_meta)
    fractionator.initial_values(0.95)

    times = {event.name: float(event.time) for event in fractionator.events}
    starts = sorted(name for name in times if "_start_" in name)
    return [
        (times[name], times.get(name.replace("_start_", "_end_"))) for name in starts
    ]


@pytest.mark.parametrize(
    "windows, expected",
    [
        pytest.param(
            [(0, 2), ISOLATED_SAMPLE, (8, 11)],
            [(8.0, 1.99)],
            id="region_wraps_cycle_end",
        ),
        pytest.param(
            # The region ends at the last sample, t = cycle time, which the
            # event reports as t = 0.
            [ISOLATED_SAMPLE, (8, 11)],
            [(8.0, 0.0)],
            id="region_ends_at_final_sample",
        ),
        pytest.param(
            [(10, 11)],
            [],
            id="single_final_sample",
        ),
    ],
)
def test_initial_values_cyclic_boundary(windows, expected):
    """Starts pair with the next end around the cycle (#459)."""
    assert _collection_windows([windows]) == [pytest.approx(pair) for pair in expected]
