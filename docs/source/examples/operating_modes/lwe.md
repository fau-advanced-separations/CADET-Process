---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
execution:
  timeout: 300

---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../')
```
(lwe_example)=
# Load, Wash, Elute
tags: single column, steric mass action law, sma, gradient

(lwe_example_concentration)=
## Concentration gradients

```{figure} ./figures/lwe_concentration_flow_sheet.svg
Flow sheet for load-wash-elute process using a single inlet.
```

```{figure} ./figures/lwe_concentration_events.svg
Events of load-wash-elute process using a single inlet and modifying its concentration.
```

```{code-cell} ipython3
:load: ../../../../examples/operating_modes/lwe_concentration.py

```

## Flow rate gradients

```{figure} ./figures/lwe_flow_rate_flow_sheet.svg
Flow sheet for load-wash-elute process using a separate inlets for buffers.
```

```{figure} ./figures/lwe_flow_rate_events.svg
Events of load-wash-elute process using multiple inlets and mofifying their flow rates.
```

```{code-cell} ipython3
:load: ../../../../examples/operating_modes/lwe_flow_rate.py

```

