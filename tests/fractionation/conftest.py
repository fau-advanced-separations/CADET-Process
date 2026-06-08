import numpy as np
import pytest
from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel.process import ProcessMeta

from tests.chromatogram_factory import gaussian_chromatogram, rectangle_chromatogram


def _make_fractionation_results(chromatograms, m_feed=None):
    """Pair chromatograms with a ProcessMeta for Fractionator construction."""
    cs = chromatograms[0].component_system
    if m_feed is None:
        m_feed = np.ones(cs.n_comp)
    process_meta = ProcessMeta(
        cycle_time=chromatograms[0].cycle_time,
        V_eluent=1.0,
        V_solid=1.0,
        m_feed=m_feed,
    )
    return chromatograms, process_meta


@pytest.fixture
def optimizer():
    return FractionationOptimizer()


@pytest.fixture
def simulation_results(request):
    """Indirect fixture: resolves a fixture name string to its value."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def single_component_wide_peak():
    cs = ComponentSystem(1)
    return _make_fractionation_results([rectangle_chromatogram(cs, [[(3, 7)]])])


@pytest.fixture
def single_component_two_peaks():
    cs = ComponentSystem(1)
    return _make_fractionation_results([rectangle_chromatogram(cs, [[(2, 4), (7, 9)]])])


@pytest.fixture
def two_components_separated():
    cs = ComponentSystem(2)
    return _make_fractionation_results(
        [rectangle_chromatogram(cs, [[(2, 4)], [(7, 9)]])],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def two_components_touching():
    cs = ComponentSystem(2)
    return _make_fractionation_results(
        [rectangle_chromatogram(cs, [[(2, 5)], [(5, 7)]])],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def two_components_overlapping():
    cs = ComponentSystem(2)
    return _make_fractionation_results(
        [rectangle_chromatogram(cs, [[(2, 5.1)], [(5, 7)]])],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def two_components_gaussian_separated():
    cs = ComponentSystem(2)
    return _make_fractionation_results(
        [gaussian_chromatogram(cs, [(3, 0.5), (7, 0.5)])],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def two_components_gaussian_close():
    cs = ComponentSystem(2)
    return _make_fractionation_results(
        [gaussian_chromatogram(cs, [(4, 1.0), (6, 1.0)])],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def two_components_fully_overlapping():
    """Both components elute at identical times; purity ceiling is 0.5."""
    cs = ComponentSystem(2)
    return _make_fractionation_results(
        [rectangle_chromatogram(cs, [[(3, 7)], [(3, 7)]])],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def two_components_separated_mass_matched():
    """Separated peaks with m_feed equal to peak integrals for recovery assertions."""
    cs = ComponentSystem(2)
    # peaks [2,4] and [7,9]: width 2, height 1.0, flow 1.0 → mass ≈ 2.0 each
    return _make_fractionation_results(
        [rectangle_chromatogram(cs, [[(2, 4)], [(7, 9)]])],
        m_feed=np.array([2.0, 2.0]),
    )


@pytest.fixture
def two_components_overlapping_mass_matched():
    """Overlapping peaks with m_feed matching integrals for recovery assertions."""
    cs = ComponentSystem(2)
    # comp0: [2, 5.1] width ≈ 3.1; comp1: [5, 7] width 2.0
    return _make_fractionation_results(
        [rectangle_chromatogram(cs, [[(2, 5.1)], [(5, 7)]])],
        m_feed=np.array([3.1, 2.0]),
    )


@pytest.fixture
def two_outlets_two_components():
    """Two separate outlets, each carrying the same 2-component separation."""
    cs = ComponentSystem(2)
    chrom_a = rectangle_chromatogram(cs, [[(2, 4)], [(7, 9)]])
    chrom_b = rectangle_chromatogram(cs, [[(2, 4)], [(7, 9)]], name="outlet_b")
    return _make_fractionation_results(
        [chrom_a, chrom_b],
        m_feed=np.array([1.0, 1.0]),
    )


@pytest.fixture
def three_components_gaussian_separated():
    cs = ComponentSystem(3)
    return _make_fractionation_results(
        [gaussian_chromatogram(cs, [(2.5, 0.4), (5, 0.4), (7.5, 0.4)])],
        m_feed=np.array([1.0, 1.0, 1.0]),
    )
