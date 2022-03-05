---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(fractionation_tutorial)=
# Chemical reactions

Let's assume the following system:

```Python3
# Components
# 0: H+
# 1: NH4+
# 2: NH3
# 3: Lys2+
# 4: Lys+
# 5: Lys
# 6: Lys-
```

Then we can now set up a reaction system:

```Python3
from reactions import ReactionSystem
r = ReactionSystem(n_comp=7)
```

First, we define the reactions in bulk phase

```Python3
# 0: NH4+(aq) <=> NH3(aq) + H+(aq)
# 1: Lys2+(aq) <=> Lys+(aq) + H+(aq)
# 2: Lys+(aq) <=> Lys(aq) + H+(aq)
# 3: Lys(aq) <=> Lys-(aq) + H+(aq)
```

We can use the `add_bulk_reaction()` function to add the reaction to the system. It takes the component `indices`, the stoichiometric `coefficients`, as well as the reaction rate as arguments. If no backward reaction is given, k_fwd is assumed to be the equilibrium constant. The rates then are scaled up by a predefined factor. Moreover, we can set `add_pore_reaction`, s.t. the same reaction is later automatically added to the particle pores.

```Python3
r.add_bulk_reaction(
    indices=[1, 2, 0],
    coefficients=[-1, 1, 1], 
    k_fwd=10**(-9.2), 
    add_pore_reaction=True 
    )
r.add_bulk_reaction([3, 4, 0], [-1, 1, 1], 10**(-2.20), add_pore_reaction=True)
r.add_bulk_reaction([4, 5, 0], [-1, 1, 1], 10**(-8.90), add_pore_reaction=True)
r.add_bulk_reaction([5, 6, 0], [-1, 1, 1], 10**(-10.28), add_pore_reaction=True)
```

Defining reactions becomes especially tricky if they span different phases. Let's assume that we want to model some kind of Adsorption process. The reactions might look something like this:

```Python3
# 0: NH4+(aq) + H+(s) <=> NH4+(s) + H+(aq)
# 1: Lys2+(aq) + H+(s) <=> Lys2+(s) + H+(aq)
# 2: Lys+(aq) + H+(s) <=> Lys+(s) + H+(aq)
```

Now we can make use of the `add_cross_phase_reaction` function. In addition to the previous function, it also takes a phases argument which assumes the index 0 for pores, and 1 for the solid phase

```Python3
r.add_cross_phase_reaction(
    indices=[1, 0, 1, 0], 
    coefficients=[-1, -1, 1, 1], 
    phases=[0, 1, 0, 1], 
    k_fwd=1.5)
r.add_cross_phase_reaction([3, 0, 3, 0], [-1, -1, 1, 1], [0, 1, 0, 1], 5)
r.add_cross_phase_reaction([4, 0, 4, 0], [-1, -1, 1, 1], [0, 1, 0, 1], 0.75)
```

Now, we can simply query the composed matrices for stoichiometry, exponent modifiers etc.
e.g.
```
>>> r.stoich_bulk
[[ 1.  1.  1.  1.]
 [-1.  0.  0.  0.]
 [ 1.  0.  0.  0.]
 [ 0. -1.  0.  0.]
 [ 0.  1. -1.  0.]
 [ 0.  0.  1. -1.]
 [ 0.  0.  0.  1.]]

>>> r.stoich_pore
[[ 1.  1.  1.  1.  0.  0.  0.]
 [-1.  0.  0.  0.  1.  0.  0.]
 [ 1.  0.  0.  0.  0.  0.  0.]
 [ 0. -1.  0.  0.  0.  1.  0.]
 [ 0.  1. -1.  0.  0.  0.  1.]
 [ 0.  0.  1. -1.  0.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.  0.]]

>>> r.stoich_solid
[[1. 1. 1.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

```

or

```
>>> r.solid_fwd_modpore
[[0. 0. 0.]
 [1. 0. 0.]
 [0. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]
 [0. 0. 0.]]
```

etc.

I will run some more tests and will publish it somewhere sometime. Just needed to share! ;-)
