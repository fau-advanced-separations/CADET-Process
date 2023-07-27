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
sys.path.append('../../../../')
```

(parallel_evaluation_guide)=
# Parallel Evaluation
**CADET-Process** provides a versatile and user-friendly framework for solving optimization problems.
One of the key features of **CADET-Process** is its ability to leverage parallelization, allowing users to exploit the full potential of their multi-core processors and significantly speed up the optimization process.

In this tutorial, the parallelization backend module of **CADET-Process** is introduced.
This module allows choosing between different parallelization strategies, each tailored to suit different hardware configuration and optimization needs.
By selecting an appropriate backend, the workload can efficiently be distributed across multiple cores, reducing the time required for function evaluations and improving overall optimization performance.

The parallelization backend module consists of four classes:
- {class}`~CADETProcess.optimization.ParallelizationBackendBase`: This class serves as the base for all parallelization backend adapters.
  It provides common attributes and methods that any parallelization backend must implement, such as {attr}`~CADETProcess.optimization.ParallelizationBase.n_cores`, the number of cores to be used for parallelization.
- {class}`~CADETProcess.optimization.SequentialBackend`: This backend is designed for cases where parallel execution is not desired or not possible.
  It evaluates the target function sequentially, one individual at a time, making it useful for single-core processors or when parallelization is not beneficial.
- {class}`~CADETProcess.optimization.Joblib`, and {class}`~CADETProcess.optimization.Pathos`: Parallel backends allowing users to leverage multiple cores efficiently for function evaluations.

All backends implement a {meth}`~CADETProcess.optimization.ParallelizationBackendBase.evaluate` method which takes a function (callable) and a population (Iterable) as input.
This method maps the provided function over the elements of the population array and returns a list containing the results of the function evaluations for each element.

To demonstrate how to use parallel backends, consider the following example.


```{code-cell} ipython3
# Import the available backends
from CADETProcess.optimization import SequentialBackend, Joblib

def square(x):
    """Simple function that returns the square of a given number."""
    return x ** 2

# Example 1: Using SequentialBackend
print("Example 1: Using SequentialBackend")
backend = SequentialBackend()
result = backend.evaluate(square, [1, 2, 3, 4])
print(result)  # Output: [1, 4, 9, 16]

# Example 2: Using Joblib Backend
print("Example 2: Using Joblib Backend")
backend = Joblib(n_cores=2)  # Specify the number of cores to be used
result = backend.evaluate(square, [1, 2, 3, 4])
print(result)  # Output: [1, 4, 9, 16]
```

All optimzers can be associated with a parallel backend.
For example, {class}`~CADETProcess.optimization.U_NSGA3` can be configured to use a parallel backend as shown below:

```{code-cell} ipython3
# Import the available backends
from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
optimizer.backend = Joblib(n_cores=2)
```
