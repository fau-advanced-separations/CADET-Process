from CADETProcess.dataStructure import ParametersGroup
from CADETProcess.dataStructure import (
    Bool, Switch, 
    RangedInteger, UnsignedInteger, UnsignedFloat,
    DependentlySizedRangedList
)

class DiscretizationParametersBase(ParametersGroup):
    _dimensionality = []
    
    def __init__(self):
        self.weno_parameters = WenoParameters()
        self.consistency_solver = ConsistencySolverParametersGroup()
        
    @property
    def dimensionality(self):
        dim = {}
        for d in self._dimensionality:
            v = getattr(self, d)
            if v is None:
                continue
            dim[d] = v
        
        return dim    

    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        parameters = super().parameters
        parameters['weno'] = self.weno_parameters.parameters
        parameters['consistency_solver'] = self.consistency_solver.parameters
        
        return parameters
    
    @parameters.setter
    def parameters(self, parameters):
        try:
            self.weno_parameters.parameters = parameters.pop('weno')
        except KeyError:
            pass
        try:
            self.consistency_solver.parameters = parameters.pop('consistency_solver')
        except KeyError:
            pass        
        
        super(DiscretizationParametersBase, self.__class__).parameters.fset(self, parameters)
        
class NoDiscretization(DiscretizationParametersBase):
    pass

class LRMDiscretizationFV(DiscretizationParametersBase):
    ncol = UnsignedInteger(default=100)
    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])
    
    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'use_analytic_jacobian', 'reconstruction',
    ]
    _dimensionality = ['ncol']

class LRMPDiscretizationFV(DiscretizationParametersBase):
    ncol = UnsignedInteger(default=100)

    par_geom = Switch(
        default='SPHERE', 
        valid=['SPHERE', 'CYLINDER', 'SLAB']
    )

    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])
    
    gs_type = Bool(default=True)
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1.0e-8)
    
    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'par_geom', 
        'use_analytic_jacobian', 'reconstruction', 
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety'
    ]
    _dimensionality = ['ncol']
    
class GRMDiscretizationFV(DiscretizationParametersBase):
    ncol = UnsignedInteger(default=100)
    npar = UnsignedInteger(default=5)

    par_geom = Switch(
        default='SPHERE', 
        valid=['SPHERE', 'CYLINDER', 'SLAB']
    )
    par_disc_type = Switch(
        default='EQUIDISTANT_PAR', 
        valid=['EQUIDISTANT_PAR', 'EQUIVOLUME_PAR', 'USER_DEFINED_PAR']
    )
    par_disc_vector = DependentlySizedRangedList(lb=0, ub=1, dep='par_disc_vector_length')
    
    par_boundary_order = RangedInteger(lb=1, ub=2, default=2)

    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])
    
    gs_type = Bool(default=True)
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1.0e-8)

    fix_zero_surface_diffusion = Bool(default=False)
    
    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'npar', 
        'par_geom', 'par_disc_type', 'par_disc_vector', 'par_boundary_order', 
        'use_analytic_jacobian', 'reconstruction', 
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety',
        'fix_zero_surface_diffusion',
    ]
    _dimensionality = ['ncol', 'npar']
        
    @property
    def par_disc_vector_length(self):
        return self.npar + 1
    

class WenoParameters(ParametersGroup):
    """Class for defining the disrectization_weno_parameters

    Defines several parameters as UnsignedInteger, UnsignedFloat and save their
    names into a list named parameters.

    See also
    --------
    ParametersGroup
    """
    boundary_model = UnsignedInteger(default=0, ub=3)
    weno_eps = UnsignedFloat(default=1e-10)
    weno_order = UnsignedInteger(default=3, ub=3)
    _parameters = ['boundary_model', 'weno_eps', 'weno_order']

class ConsistencySolverParametersGroup(ParametersGroup):
    """Class for defining the consistency solver parameters for cadet.

    See also
    --------
    ParametersGroup
    """
    solver_name = Switch(
        default='LEVMAR', 
        valid=['LEVMAR', 'ATRN_RES', 'ARTN_ERR', 'COMPOSITE']
    )
    init_damping = UnsignedFloat(default=0.01)
    min_damping = UnsignedFloat(default=0.0001)
    max_iterations = UnsignedInteger(default=50)
    subsolvers = Switch(
        default='LEVMAR', 
        valid=['LEVMAR', 'ATRN_RES', 'ARTN_ERR']
    )

    _parameters = [
        'solver_name', 'init_damping', 'min_damping', 
        'max_iterations', 'subsolvers'
    ]    