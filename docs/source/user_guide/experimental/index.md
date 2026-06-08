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

**2. Signal calibration** ({doc}`calibration`)
Convert raw detector signals (UV absorbance, conductivity) carried by {class}`~CADETProcess.reference.ReferenceIO` objects into physical concentration units.
Steps include baseline correction, area normalization, Beer-Lambert conversion, and multi-wavelength deconvolution.

**3. System characterization** ({doc}`characterization`)
Fit model parameters (bed porosity, axial dispersion, film diffusion, adsorption constants) by minimizing the difference between simulated and measured chromatograms.
Each {class}`~CADETProcess.characterization.CharacterizeBase` subclass encodes domain knowledge about which parameters to fit for a given experiment type and supplies physically sensible default bounds.

```{toctree}
:maxdepth: 2
:hidden:

instruments
calibration
characterization
```
