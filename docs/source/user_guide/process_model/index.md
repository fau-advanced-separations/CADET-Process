(process_model_guide)=
# Process Model

Starting point of process development is the setup of the {class}`~CADETProcess.processModel.Process` (see {ref}`Framework overview <framework_overview>`) model, i.e., the specific configuration of the chromatographic process.
This is realized using {class}`UnitOperations <CADETProcess.processModel.UnitBaseClass>` as building blocks.
A {class}`UnitOperation <CADETProcess.processModel.UnitBaseClass>` represents the physico-chemical behavior of an apparatus and holds the model parameters.
For more information refer to {ref}`unit_operation_guide`.

All {class}`UnitOperations <CADETProcess.processModel.UnitBaseClass>` can be associated with {class}`BindingModels <CADETProcess.processModel.BindingBaseClass>` that describe the interaction of components with surfaces or chromatographic stationary phases.
For this purpose, a variety of equilibrium relations, for example,the simple {class}`~CADETProcess.processModel.Linear` adsorption isotherm, competitive forms of the {class}`~CADETProcess.processModel.Langmuir` and the {class}`~CADETProcess.processModel.BiLangmuir` models, as well as the competitive {class}`~CADETProcess.processModel.StericMassAction` law can be selected.
For more information refer to {ref}`binding_models_guide`.

Moreover, {class}`ReactionModels <CADETProcess.processModel.ReactionBaseClass>` can be used to model chemical reactions.
For more information refer to {ref}`reaction_models_guide`.

Multiple {class}`UnitOperation <CADETProcess.processModel.UnitBaseClass>` can be connected in a {class}`~CADETProcess.processModel.FlowSheet` which describes the mass transfer between the individual units.
For more information refer to {ref}`flow_sheet_guide`.

Finally, dynamic {class}`Events <CADETProcess.dynamicEvents.Event>` can be defined to model dynamic changes of model parameters, including flow rates system connectivity.
For more information refer to {ref}`process_guide`.

In the following, the different modules are introduced.


```{toctree}
:maxdepth: 2

component_system
binding_model
reaction
unit_operation
flow_sheet
process
```
