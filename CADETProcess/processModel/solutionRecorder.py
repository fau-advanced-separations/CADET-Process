from CADETProcess.dataStructure import Bool, ParametersGroup


class SolutionRecorderBase(ParametersGroup):
    pass


class SolutionRecorderIO(SolutionRecorderBase):
    """Select which simulation solution to store."""
    write_solution_last_unit = Bool(default=False)
    write_coordinates = Bool(default=True)

    write_solution_inlet = Bool(default=True)
    write_soldot_inlet = Bool(default=False)
    write_sens_inlet = Bool(default=False)
    write_sensdot_inlet = Bool(default=False)

    write_solution_outlet = Bool(default=True)
    write_soldot_outlet = Bool(default=False)
    write_sens_outlet = Bool(default=False)
    write_sensdot_outlet = Bool(default=False)

    _parameters = [
        'write_coordinates',
        'write_solution_last_unit',

        'write_solution_inlet',
        'write_soldot_inlet',
        'write_sens_inlet',
        'write_sensdot_inlet',

        'write_solution_outlet',
        'write_soldot_outlet',
        'write_sens_outlet',
        'write_sensdot_outlet'
    ]


class SolutionRecorderBulk(SolutionRecorderBase):
    write_solution_bulk = Bool(default=False)
    write_soldot_bulk = Bool(default=False)
    write_sens_bulk = Bool(default=False)
    write_sensdot_bulk = Bool(default=False)

    _parameters = [
        'write_solution_bulk',
        'write_soldot_bulk',
        'write_sens_bulk',
        'write_sensdot_bulk',
    ]


class SolutionRecorderParticle(SolutionRecorderBase):
    write_solution_particle = Bool(default=False)
    write_soldot_particle = Bool(default=False)
    write_sens_particle = Bool(default=False)
    write_sensdot_particle = Bool(default=False)

    write_solution_solid = Bool(default=False)
    write_soldot_solid = Bool(default=False)
    write_sens_solid = Bool(default=False)
    write_sensdot_solid = Bool(default=False)

    _parameters = [
            'write_solution_particle',
            'write_soldot_particle',
            'write_sens_particle',
            'write_sensdot_particle',

            'write_solution_solid',
            'write_soldot_solid',
            'write_sens_solid',
            'write_sensdot_solid',
    ]


class SolutionRecorderFlux(SolutionRecorderBase):
    write_solution_flux = Bool(default=False)
    write_soldot_flux = Bool(default=False)
    write_sensdot_flux = Bool(default=False)
    write_sens_flux = Bool(default=False)

    _parameters = [
        'write_solution_flux',
        'write_soldot_flux',
        'write_sens_flux',
        'write_sensdot_flux',
    ]


class SolutionRecorderVolume(SolutionRecorderBase):
    write_solution_volume = Bool(default=True)
    write_soldot_volume = Bool(default=False)
    write_sens_volume = Bool(default=False)
    write_sensdot_volume = Bool(default=False)

    _parameters = [
        'write_solution_volume',
        'write_soldot_volume',
        'write_sens_volume',
        'write_sensdot_volume'
    ]


class TubularReactorRecorder(SolutionRecorderIO, SolutionRecorderBulk):
    pass


class LRMRecorder(
        SolutionRecorderIO, SolutionRecorderBulk, SolutionRecorderParticle):
    pass


class LRMPRecorder(
        SolutionRecorderIO, SolutionRecorderBulk, SolutionRecorderFlux,
        SolutionRecorderParticle, SolutionRecorderVolume):
    pass


class GRMRecorder(
        SolutionRecorderIO, SolutionRecorderBulk, SolutionRecorderFlux,
        SolutionRecorderParticle):
    pass


class CSTRRecorder(
        SolutionRecorderIO, SolutionRecorderBulk, SolutionRecorderFlux,
        SolutionRecorderParticle, SolutionRecorderVolume):
    pass
