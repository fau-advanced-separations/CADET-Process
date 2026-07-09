---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../../')
%matplotlib inline
```

(characterization_guide)=
# LC System Characterization

The {mod}`~CADETProcess.characterization` module sets up parameter estimation problems for LC systems.
Each {class}`~CADETProcess.characterization.CharacterizeBase` subclass is an {class}`~CADETProcess.optimization.OptimizationProblem` specialized for a particular experiment type, encoding which parameters are fitted, their default bounds and transforms, and any experiment-specific constraints or parameter couplings.
Users supply the process, the simulator, and the comparator; the class wires them into a ready-to-optimize problem.

**Choosing a subclass:**

- Dead volume of a single tubing segment: {class}`~CADETProcess.characterization.CharacterizeTubing`
- Pre-injection path (tubing length + mixer volume): {class}`~CADETProcess.characterization.CharacterizePreInjection`
- Packed bed (porosity + axial dispersion): {class}`~CADETProcess.characterization.CharacterizeBed`
- Particle-phase transport (film/pore diffusion, particle porosity): {class}`~CADETProcess.characterization.CharacterizeParticles`
- Binding model capacity: {class}`~CADETProcess.characterization.CharacterizeCapacity`
- SMA adsorption parameters: {class}`~CADETProcess.characterization.CharacterizeAdsorptionParameters`

```{code-cell} ipython3
import numpy as np
from CADETProcess.processModel import (
    ComponentSystem, LumpedRateModelWithPores, StericMassAction,
)
from CADETProcess.instruments import LCFlowSheet, PulseInjection, LWE
from CADETProcess.comparison import Comparator
from CADETProcess.reference import ReferenceIO
from CADETProcess.characterization import (
    setup_comparators,
    CharacterizeBed,
    CharacterizeParticles,
    CharacterizeAdsorptionParameters,
)
```


## Workflow

A typical characterization workflow has three steps.

**1. Build the process.**
Create an `LCFlowSheet` and an `LCProcess` template matching the experimental protocol.
The example below runs a pulse injection of a non-binding tracer through the packed bed to characterize its transport parameters.

```{code-cell} ipython3
cs = ComponentSystem(["Salt"])
Q = 8.3e-9  # m³/s

fs = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    ColumnModel=LumpedRateModelWithPores,
)
process = PulseInjection(
    "bed_characterization",
    fs,
    c_buffer_a=[100.0],
    c_sample=[0.0],
    cycle_time=600.0,
    flow_rate=Q,
)
```

**2. Build the comparator.**
A {class}`~CADETProcess.comparison.Comparator` holds one or more references and the difference metrics used to score the simulation against the experiment.
The `solution_path` locates the simulated signal inside `SimulationResults.solution_cycles`, e.g. the outlet concentration of a column or tubing segment.

```{code-cell} ipython3
:tags: [remove-cell]

# Synthetic "experimental" data for illustration
time = np.linspace(0, 600, 601)
solution = np.exp(-0.5 * ((time - 300) / 30) ** 2).reshape(-1, 1)
reference = ReferenceIO("uv_280", time, solution, flow_rate=Q)
reference.component_system = cs
```

```{code-cell} ipython3
comparator = Comparator("bed_characterization")
comparator.add_reference(reference)
comparator.add_difference_metric(
    "Shape",
    reference,
    solution_path="column.outlet.outlet[0]",
)
```

**3. Create the characterization problem and optimize.**
Pass the process, comparator, and simulator to the appropriate subclass.
The result is an {class}`~CADETProcess.optimization.OptimizationProblem` ready for any optimizer.

```{code-cell} ipython3
:tags: [skip-execution]

from CADETProcess.simulator import Cadet
from CADETProcess.optimization import U_NSGA3

simulator = Cadet()

char = CharacterizeBed("bed", process, comparator, simulator)

optimizer = U_NSGA3()
optimizer.optimize(char)
```


## Optimization structure

Every `Characterize*` instance is an ordinary {class}`~CADETProcess.optimization.OptimizationProblem`.

- **Variables** correspond to the selected physical parameters with predefined bounds and transforms.
- **Objectives** are provided by the comparator metrics; each comparator contributes its own objective (or objectives if the comparator has multiple metrics).
- **Multi-process characterization** creates one objective per process while all processes share the same variable set.
Each evaluation automatically generates comparison plots in the optimizer callback directory.


## Subclasses


### CharacterizeTubing

Fits length $l \in [0.01, 2.0]$ m and axial dispersion $D_{ax} \in [10^{-9}, 10^{-2}]$ m²/s of a named tubing segment.
The `tubing` argument must match the unit name in the flow sheet.

```{code-cell} ipython3
:tags: [skip-execution]

from CADETProcess.characterization import CharacterizeTubing

char = CharacterizeTubing(
    "tubing", process, "tubing_post_column", comparator, simulator
)
print(char.variable_names)
# ['tubing_post_column_length', 'tubing_post_column_axial_dispersion']
```


### CharacterizePreInjection

Fits `tubing_pre_injection` length $\in [0.01, 2.0]$ m and mixer volume $\in [10^{-8}, 10^{-5}]$ m³.

```{code-cell} ipython3
:tags: [skip-execution]

from CADETProcess.characterization import CharacterizePreInjection

char = CharacterizePreInjection("pre_inj", process, comparator, simulator)
print(char.variable_names)
# ['tubing_pre_injection_length', 'mixer_volume']
```


### CharacterizeBed

Fits column bed porosity $\varepsilon_c \in [0.2, 0.6]$ and axial dispersion $D_{ax} \in [10^{-11}, 10^{-3}]$ m²/s.
Run a pulse injection with a non-binding tracer so that peak broadening reflects hydrodynamic transport rather than adsorption.

```{code-cell} ipython3
:tags: [skip-execution]

char = CharacterizeBed("bed", process, comparator, simulator)
print(char.variable_names)
# ['bed_porosity', 'axial_dispersion']
```


### CharacterizeParticles

Fits a selectable combination of particle-phase transport parameters for a non-binding tracer.
At least one flag must be set.
In practice, fitting many transport parameters simultaneously requires sufficiently informative experiments; start with the parameters most likely to dominate the observed peak shape.
`pore_diffusion` is only meaningful for {class}`~CADETProcess.processModel.GeneralRateModel` columns; for {class}`~CADETProcess.processModel.LumpedRateModelWithPores`, only the remaining parameters should be fitted.

```{code-cell} ipython3
:tags: [skip-execution]

from CADETProcess.characterization import CharacterizeParticles

char = CharacterizeParticles(
    "particles", process, comparator, simulator,
    include_particle_porosity=True,
    include_film_diffusion=True,
    component_index=0,
)
print(char.variable_names)
# ['particle_porosity', 'film_diffusion']
```


### CharacterizeCapacity

Fits binding model capacity $q_{max} \in [0.01, 2.0]$ mol/m³.
The binding model attached to the column must expose a `capacity` parameter, such as {class}`~CADETProcess.processModel.StericMassAction`.

```{code-cell} ipython3
:tags: [skip-execution]

from CADETProcess.characterization import CharacterizeCapacity

char = CharacterizeCapacity("capacity", process, comparator, simulator)
print(char.variable_names)
# ['capacity']
```


### CharacterizeAdsorptionParameters

Fits {class}`~CADETProcess.processModel.StericMassAction` adsorption parameters from gradient elution data.
Constructing the object modifies the attached binding model configuration to match the selected kinetic mode, so the process passed in is mutated in place.
Two modes are available.

**Rapid-equilibrium mode** (`is_kinetic=False`, default):
`desorption_rate` is fixed to 1, so `adsorption_rate` numerically equals the equilibrium constant.
Two free parameters: `characteristic_charge` and `adsorption_rate`.

**Kinetic mode** (`is_kinetic=True`):
two optimization variables (`equilibrium_constant` and `kinetic_constant`) drive `adsorption_rate`
and `desorption_rate` as dependent variables through the relations
$k_\text{ads} = K_\text{eq} / k_\text{kin}$ and $k_\text{des} = 1 / k_\text{kin}$,
which enforces $K_\text{eq} = k_\text{ads} / k_\text{des}$ by construction.
A Pareto-front selector picks the best individual by summing objectives.

```{code-cell} ipython3
:tags: [skip-execution]

from CADETProcess.processModel import StericMassAction
from CADETProcess.characterization import CharacterizeAdsorptionParameters

cs2 = ComponentSystem(["Salt", "Protein"])
fs2 = LCFlowSheet(
    cs2,
    sample_loop_volume=50e-9,
    ColumnModel=LumpedRateModelWithPores,
    BindingModel=StericMassAction,
)
lwe = LWE(
    "lwe", fs2,
    c_buffer_a=[20.0, 0.0], c_buffer_b=[1000.0, 0.0], c_sample=[20.0, 0.5],
    delta_t_wash=120.0, delta_t_elute=600.0, delta_t_final_wash=120.0,
    flow_rate_wash=Q,
)

char = CharacterizeAdsorptionParameters(
    "sma", lwe, comparator, simulator,
    is_kinetic=False,
    component_index=1,
)
print(char.variable_names)
# ['characteristic_charge', 'adsorption_rate']
```


## Multi-process example

When the same characterization is run across several gradient lengths simultaneously, pass a list of processes and a matching list of comparators.
Rather than build each comparator by hand as in step 2 of the workflow, {func}`~CADETProcess.characterization.setup_comparators` builds the whole list at once: one {class}`~CADETProcess.comparison.Comparator` per process, all sharing the same `solution_path` and difference metrics.
`start` and `end` accept either a scalar, applied to every process, or a list with one value per process, so each gradient can have its own peak window.
For heterogeneous comparisons, such as UV and conductivity channels on the same process, build the comparators directly instead.

```{code-cell} ipython3
:tags: [skip-execution]

processes = [setup_lwe(cv) for cv in [4, 8, 12, 16]]  # one LWE per gradient length
references = [load_reference(cv) for cv in [4, 8, 12, 16]]
start_times = [peak_start[cv] for cv in [4, 8, 12, 16]]
end_times   = [peak_end[cv]   for cv in [4, 8, 12, 16]]

comparators = setup_comparators(
    processes, references,
    solution_path="tubing_post_column.outlet",
    metrics=["Shape"],
    components=["Protein"],
    start=start_times,
    end=end_times,
)

char = CharacterizeAdsorptionParameters(
    "sma_multi", processes, comparators, simulator,
    component_index=1,
)
```

The optimization problem has one objective per process (one per gradient), all sharing the same parameter variables.
The optimizer searches for a parameter vector that jointly improves agreement across all gradients.
