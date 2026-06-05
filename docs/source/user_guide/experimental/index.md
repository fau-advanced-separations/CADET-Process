---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(experimental_workflows)=
# Experimental Workflows

The pages in this chapter cover the pipeline from raw instrument data to fitted model parameters.
The three steps are independent and can be applied in any combination, but they follow a natural sequence:

**1. Instrument setup** ({doc}`instruments`)
Model the LC system as a flow sheet, define the experiment protocol using a process template, and generate synthetic data if needed before real experiments are available.

```{toctree}
:maxdepth: 2
:hidden:

instruments
```
