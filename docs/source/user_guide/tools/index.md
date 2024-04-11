(tools_guide)=
# Tools

**CADET-Process** provides various tools that can be used for pre- and postprocessing.

## Model Builder

The {mod}`~CADETProcess.modelBuilder` module offers classes that simplify the setup of compartment models used in bioreactors and carousel/SMB systems, where multiple columns are used, and their positions are changed dynamically during operation.

```{toctree}
:maxdepth: 2

carousel_builder
smb_design
compartment_builder
```

## Buffer equlibria and pH

The {mod}`~CADETProcess.equilibria` module contains methods to calculate buffer capacity and simulate pH using deprotonation reactions.
These tools are essential for the optimization of bioprocesses and the design of chromatographic separation processes, where accurate pH control is critical for efficient separation.

```{toctree}
:maxdepth: 2

deprotonation_reactions
buffer_capacity
```
