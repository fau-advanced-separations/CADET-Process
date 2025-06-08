from CADETProcess.dataStructure import Bool, Structure


class BaseMixin(Structure):
    """
    Recorder for last state and coordinates.

    Attributes
    ----------
    write_solution_last_unit : bool, optional
        If True, write final state of the unit operation. The default is False.
    write_coordinates : bool, optional
        If True, write coordinates of the unit operation. The default is True.
    """

    write_solution_last_unit = Bool(default=False)
    write_coordinates = Bool(default=True)

    _parameters = [
        "write_coordinates",
        "write_solution_last_unit",
    ]


class IOMixin(Structure):
    """
    Recorder for inlet and outlet streams.

    Attributes
    ----------
    write_solution_inlet : bool, optional
        If True, write solution of the unit operation inlet. The default is True.
    write_soldot_inlet : bool, optional
        If True, write derivative of the unit operation inlet. The default is False.
    write_sens_inlet : bool, optional
        If True, write sensitivities of the unit operation inlet. The default is True.
    write_sensdot_inlet : bool, optional
        If True, write sensitivities derivatives of the unit operation inlet.
        The default is False.
    write_solution_outlet : bool, optional
        If True, write solution of the unit operation outlet. The default is True.
    write_soldot_outlet : bool, optional
        If True, write derivative of the unit operation outlet. The default is False.
    write_sens_outlet : bool, optional
        If True, write sensitivities of the unit operation outlet. The default is True.
    write_sensdot_outlet : bool, optional
        If True, write sensitivities derivatives of the unit operation outlet.
        The default is False.
    """

    write_solution_inlet = Bool(default=True)
    write_soldot_inlet = Bool(default=False)
    write_sens_inlet = Bool(default=True)
    write_sensdot_inlet = Bool(default=False)

    write_solution_outlet = Bool(default=True)
    write_soldot_outlet = Bool(default=False)
    write_sens_outlet = Bool(default=True)
    write_sensdot_outlet = Bool(default=False)

    _parameters = [
        "write_solution_inlet",
        "write_soldot_inlet",
        "write_sens_inlet",
        "write_sensdot_inlet",
        "write_solution_outlet",
        "write_soldot_outlet",
        "write_sens_outlet",
        "write_sensdot_outlet",
    ]


class BulkMixin(Structure):
    """
    Recorder for bulk solution.

    Attributes
    ----------
    write_solution_bulk : bool, optional
        If True, write solution of the bulk. The default is False.
    write_soldot_bulk : bool, optional
        If True, write derivative of the bulk. The default is False.
    write_sens_bulk : bool, optional
        If True, write sensitivities of the bulk. The default is False.
    write_sensdot_bulk : bool, optional
        If True, write sensitivities derivatives of the bulk. The default is False.
    """

    write_solution_bulk = Bool(default=False)
    write_soldot_bulk = Bool(default=False)
    write_sens_bulk = Bool(default=False)
    write_sensdot_bulk = Bool(default=False)

    _parameters = [
        "write_solution_bulk",
        "write_soldot_bulk",
        "write_sens_bulk",
        "write_sensdot_bulk",
    ]


class ParticleMixin(Structure):
    """
    Recorder for particle liquid phase solution.

    Attributes
    ----------
    write_solution_particle : bool, optional
        If True, write the solution of the particle liquid phase. The default is False.
    write_soldot_particle : bool, optional
        If True, write the derivative of the particle liquid phase.
        The default is False.
    write_sens_particle : bool, optional
        If True, write the sensitivities of the particle liquid phase.
        The default is False.
    write_sensdot_particle : bool, optional
        If True, write the sensitivities derivatives of the particle liquid phase.
        The default is False.
    """

    write_solution_particle = Bool(default=False)
    write_soldot_particle = Bool(default=False)
    write_sens_particle = Bool(default=False)
    write_sensdot_particle = Bool(default=False)

    _parameters = [
        "write_solution_particle",
        "write_soldot_particle",
        "write_sens_particle",
        "write_sensdot_particle",
    ]


class SolidMixin(Structure):
    """
    Recorder for particle solid phase solution.

    Attributes
    ----------
    write_solution_solid : bool, optional
        If True, write the solution of the particle solid phase. The default is False.
    write_soldot_solid : bool, optional
        If True, write the derivative of the particle solid phase. The default is False.
    write_sens_solid : bool, optional
        If True, write the sensitivities of the particle solid phase.
        The default is False.
    write_sensdot_solid : bool, optional
        If True, write the sensitivities derivatives of the particle solid phase.
        The default is False.
    """

    write_solution_solid = Bool(default=False)
    write_soldot_solid = Bool(default=False)
    write_sens_solid = Bool(default=False)
    write_sensdot_solid = Bool(default=False)

    _parameters = [
        "write_solution_solid",
        "write_soldot_solid",
        "write_sens_solid",
        "write_sensdot_solid",
    ]


class FluxMixin(Structure):
    """
    Recorder for flux solution.

    Attributes
    ----------
    write_solution_flux : bool, optional
        If True, write the solution of the flux. The default is False.
    write_soldot_flux : bool, optional
        If True, write the derivative of the flux. The default is False.
    write_sens_flux : bool, optional
        If True, write the sensitivities of the flux. The default is False.
    write_sensdot_flux : bool, optional
        If True, write the sensitivities derivatives of the flux. The default is False.
    """

    write_solution_flux = Bool(default=False)
    write_soldot_flux = Bool(default=False)
    write_sensdot_flux = Bool(default=False)
    write_sens_flux = Bool(default=False)

    _parameters = [
        "write_solution_flux",
        "write_soldot_flux",
        "write_sens_flux",
        "write_sensdot_flux",
    ]


class VolumeMixin(Structure):
    """
    Recorder for unit volume solution.

    Attributes
    ----------
    write_solution_volume : bool, optional
        If True, write the solution of the unit volume. The default is True.
    write_soldot_volume : bool, optional
        If True, write the derivative of the unit volume. The default is False.
    write_sens_volume : bool, optional
        If True, write the sensitivities of the unit volume. The default is False.
    write_sensdot_volume : bool, optional
        If True, write the sensitivities derivatives of the unit volume.
        The default is False.
    """

    write_solution_volume = Bool(default=True)
    write_soldot_volume = Bool(default=False)
    write_sens_volume = Bool(default=False)
    write_sensdot_volume = Bool(default=False)

    _parameters = [
        "write_solution_volume",
        "write_soldot_volume",
        "write_sens_volume",
        "write_sensdot_volume",
    ]


class SolutionRecorderBase:
    """Base class for solution recorders."""

    pass


class IORecorder(SolutionRecorderBase, BaseMixin, IOMixin):
    """
    Recorder for inlets and outlets.

    See Also
    --------
    BaseMixin
    IOMixin
    CADETProcess.processModel.Inlet
    CADETProcess.processModel.Outlet
    """

    pass


class TubularReactorRecorder(SolutionRecorderBase, BaseMixin, IOMixin, BulkMixin):
    """
    Recorder for TubularReactor.

    See Also
    --------
    BaseMixin
    IOMixin
    SolutionRecorderBulk
    CADETProcess.processModel.TubularReactor
    """

    pass


class LRMRecorder(SolutionRecorderBase, BaseMixin, IOMixin, BulkMixin, SolidMixin):
    """
    Recorder for TubularReactor.

    See Also
    --------
    BaseMixin
    IOMixin
    BulkMixin
    SolidMixin
    CADETProcess.processModel.LumpedRateModelWithoutPores
    """

    pass


class LRMPRecorder(
    SolutionRecorderBase,
    BaseMixin,
    IOMixin,
    BulkMixin,
    FluxMixin,
    ParticleMixin,
    SolidMixin,
):
    """
    Recorder for TubularReactor.

    See Also
    --------
    BaseMixin
    IOMixin
    BulkMixin
    FluxMixin
    ParticleMixin
    SolidMixin
    CADETProcess.processModel.LumpedRateModelWithPores
    """

    pass


class GRMRecorder(
    SolutionRecorderBase,
    BaseMixin,
    IOMixin,
    BulkMixin,
    FluxMixin,
    ParticleMixin,
    SolidMixin,
):
    """
    Recorder for TubularReactor.

    See Also
    --------
    BaseMixin
    IOMixin
    BulkMixin
    FluxMixin
    ParticleMixin
    SolidMixin
    CADETProcess.processModel.GeneralRateModel
    """

    pass


class CSTRRecorder(
    SolutionRecorderBase, BaseMixin, IOMixin, BulkMixin, SolidMixin, VolumeMixin
):
    """
    Recorder for TubularReactor.

    See Also
    --------
    BaseMixin
    IOMixin
    BulkMixin
    SolidMixin
    VolumeMixin
    CADETProcess.processModel.Cstr
    """

    pass


class MCTRecorder(SolutionRecorderBase, BaseMixin, IOMixin, BulkMixin):
    """
    Recorder for TubularReactor.

    See Also
    --------
    BaseMixin
    IOMixin
    BulkMixin
    CADETProcess.processModel.MCT
    """

    pass
