---
jupytext:
  formats: md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chrom_system_characterization)=
# Characterization of Chromatographic System: Modeling Periphery, Transport, and Binding Parameters

To accurately model a chromatographic system in **CADET-Process**, several steps are necessary.
First, the system periphery, including components such as tubing and valves, must be characterized to account for any dead volumes that may cause time shifts in the signal, as well as dispersion that can lead to peak broadening
Next, column parameters such as porosities and axial dispersion need to be determined.
Finally, the binding model parameters must be characterized, as they determine the separation of components.

In the following sections, the steps required to model a chromatographic system in **CADET-Process** are outlined, including how to fit the model to experimental data.

```{toctree}
:maxdepth: 1

system_periphery
column_transport_parameters
particle_porosity
binding_model_parameters
```
