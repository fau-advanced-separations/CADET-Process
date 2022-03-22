# Create inlet profile from experimental data

```{code-block} python
import numpy as np
from scipy import interpolate


time = np.linspace(0, batch_binary.cycle_time, 1001)

sig = 10
tr_1 = 1/10*batch_binary.cycle_time
c1 = np.exp(-np.power(time - tr_1, 2.) / (2 * np.power(sig, 2.))).reshape(1001,)
sig = 15
tr_2 = 2/10*batch_binary.cycle_time
c2 = np.exp(-np.power(time - tr_2, 2.) / (2 * np.power(sig, 2.))).reshape(1001,)

c = np.array([c1,c2]).T

s = 1e-3
ppoly = []
for i in c.T:
    splrep_spline = interpolate.splrep(time, i, s=s)
    ppoly.append(interpolate.PPoly.from_spline(splrep_spline))

c_poly = np.array([p(time) for p in ppoly]).T

import matplotlib.pyplot as plt
plt.figure()
plt.plot(time, c, label='original')
plt.plot(time, c_poly, label=f'ppoly {s}')
plt.legend()
```

```{code-block} python
from examples.forward_simulation.batch_binary import batch_binary, feed
batch_binary.add_inlet_profile(feed, time, c)
batch_binary.cycle_time *= 3

if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()

    batch_binary_sim_results = process_simulator.simulate(batch_binary)


```
