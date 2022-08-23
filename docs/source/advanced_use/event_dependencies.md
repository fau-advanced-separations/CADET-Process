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
sys.path.append('../../../')
```

$$\require{mhchem}$$

(event_dependencies_tutorial)=
# Event Dependencies
In order to reduce the complexity of process configurations, `Event` dependencies can be specified that define the time at which an `Event` occurs as a function of other `Event` times.
Especially for more advanced processes, this reduces the degrees of freedom and facilitates the overall handiness.
For this purpose, `Durations` can also be defined to describe the time between the execution of two `Events`.

The time of a dependent event is calculated using the following equation

$$
t = \sum_i^{n_{dep}} \lambda_i \cdot f_i(t_{dep,i}),
$$

where $n_{dep}$ denotes the number of dependencies of a given `Event`, $\lambda_i$ denotes a linear factor, and $f_i$ is some transform function.

To add a dependency in **CADET-Process**, use the `add_event_dependency` method of the `Process` class.

```{eval-rst}
.. currentmodule:: CADETProcess.dynamicEvents.event
.. automethod:: EventHandler.add_event_dependency
```

Alternatively, the dependencies can also already be added in the `add_event` method when creating the `Event` in the first place.

```{eval-rst}
.. currentmodule:: CADETProcess.dynamicEvents.event
.. automethod:: EventHandler.add_event
```

## Simple Dependency
Consider the batch elution process (see {ref}`here <batch_elution_example>`).

```{figure} ../../../examples/operating_modes/figures/batch_elution_events_simple.svg
Events of batch elution process.
```

Here, every time the feed is switched on, the elution buffer should be switched off and vice versa.


```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.batch_elution import flow_sheet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'batch elution')
```

First, the independent events are added:
```{code-cell} ipython3
Q = 60e-6/60
_ = process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
_ = process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
```

Then, the dependencies are set.
As previously mentioned, one option is to first create an event and then add the dependency:

```{code-cell} ipython3
_ = process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
_ = process.add_event_dependency('eluent_off', ['feed_on'])
```

Alternatively, the dependency can be set when creating the event:

```{code-cell} ipython3
process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q, dependencies=['feed_off'])
```

Now, only one of the event times needs to be adjusted, e.g. in a process optimzation setting.

```{code-cell} ipython3
process.feed_off.time = 10
print(f'feed_off: {process.feed_off.time}')
print(f'eluent_on: {process.eluent_on.time}')
```

## Complex Dependency
Another, more complex scenario can be applied for the {ref}`SSR process <ssr_example>`.

```{figure} ../../../examples/operating_modes/figures/ssr_events.svg
Events of SSR process.
```

Here, the injection time is defined as:

$$
\Delta t_{inject} = \Delta t_{feed} + \Delta t_{recycle}.
$$

Consequently, the end of the injection is defined as:

$$
t_{inject, off} = t_{inject, on} + \Delta t_{feed} + \Delta t_{recycle} = t_{inject, on} + \Delta t_{feed} + t_{recycle, off} - t_{recycle, on}.
$$

Furthermore, fresh feed should only be added to the tank after injection has finished.
To model this system such that feed duration and recycle times can be modified independently, multiple dependencies are required, as well as the definition of a `Duration` object.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.steady_state_recycling import flow_sheet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'ssr')
```

First, the start of the injection is defined to occur at $t=0$.

```{code-cell} ipython3
_ = process.add_event('inject_on', 'flow_sheet.tank.flow_rate', Q, time=0)
_ = process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0, time=0)
```

Also, the recycle times are considered independent events.

```{code-cell} ipython3
_ = process.add_event('recycle_on', 'flow_sheet.output_states.column', 0, time=360)
_ = process.add_event('recycle_off', 'flow_sheet.output_states.column', 1, time=420)
```

Because the time of the feed changes depending on the recycle times, a feed duration is added to the process.
Note that `Durations` do not actually perform any changes to the system.
They purely serve as a dummy events for facilitating the setup of such systems.

```{code-cell} ipython3
process.add_duration('feed_duration', 60)
```

Now, the `inject_off` event can be added:

```{code-cell} ipython3
_ = process.add_event('inject_off', 'flow_sheet.tank.flow_rate', 0.0)
_ = process.add_event_dependency(
    'inject_off', ['inject_on', 'feed_duration', 'recycle_off', 'recycle_on'],
    [1, 1, 1, -1]
)
```

Finally, the rest of the dependent events can be added.
Note that events can also depend on the time of other dependent events.
However, it is important to avoid circular dependencies, i.e. when one `Event` depends on itself.

```{code-cell} ipython3
_ = process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q, dependencies=['inject_off'])
_ = process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q, dependencies=['inject_off'])
```

With this setup, the feed duration and recycle times can be optimized independently while the rest of the event times are automatically determined.

