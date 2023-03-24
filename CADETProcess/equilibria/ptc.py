"""Pseudo-transient continuation for solving nonlinear equation systems.

The main method is ptc, a pseudo-transient continuation method with
switched evolution relaxation.

This code was written by Samuel Leweke (University of Cologne) in 2020.

"""

__author__ = "Samuel Leweke"
__contact__ = "leweke@math.uni-koeln.de"
__copyright__ = "Copyright 2020, University of Cologne"


import numpy as np
import scipy.linalg
import math


def scaled_norm2(x, scale):
    return math.sqrt(np.sum(np.square(x / scale)) / len(x))


def norm2(x):
    return math.sqrt(np.sum(np.square(x)) / len(x))


def jacrow_scale(jacMat, scale):
    return jacMat / scale[:, None]


def ptc(
        x, f, jacF, tau, tol,
        scale=None, maxIter=50, maxNonMonotone=5,
        quiet=True, variant=False):
    r"""Solve a nonlinear equation system using pseudo-transient continuation.

    The nonlinear equation system f(x) = 0 is solved using pseudo-transient
    continuation (PTC), which introduce pseudo time and computes the steady
    state of the ODE dx / dt = f(x).

    After resolving the transient regime at the beginning of the time
    integration, PTC quickly increases the time step size to arrive at the
    equilibrium. It can be interpreted as a linear-implicit Euler scheme for
    the ODE, which tends to a Newton method as the time step size increases.

    The step size is adaptively determined by switched evolution relaxation
    (SER).

    See Deuflhard, Peter (2011). Newton Methods for Nonlinear Problems
    (Vol. 35). Berlin, Heidelberg: Springer.
    http://doi.org/10.1007/978-3-642-23899-4


    Parameters
    ----------
    x : list
        Initial point as numpy array.
    f : callable
        Function with signature f(x) that returns the residual.
    jacF : callable
        Function with signature jacF(x) that returns the Jacobian of f.
    tau : TYPE
        Initial pseudo time step size.
    tol : TYPE
        Target tolerance for scaled root mean square residual.
    scale : np.array, optional
        (positive) diagonal scaling coefficients.
        The scaled root mean square norm is given by
        .. math::
            || x || = \sqrt{ (1/n) * \sum (x_i / v_i)^2}
        If None, no scaling is applied. The default is None.
    maxIter : int, optional
        Maximum number of iterations. The default is 50.
    maxNonMonotone : int, optional
        Maximum number of iterations with non-decreasing residual in a row.
        The default is 5.
    quiet : bool, optional
        If True, norm of step, scaled root mean square residual, and step size
        are printed on each iteration. The default is True.
    variant : bool, optional
        If True, (1/tau * I - F'(x)) * dx = F(x) is solved.
        If False, (I - tau * F'(x)) * dx = F(x) is solved.
        The default is False

    Returns
    -------
    status : int
        -2: Failed due to non-decreasing residual
        -1: Failed due to singular Jacobian
         0: Converged with passing residual test
         1: Exceeded maximum number of iterations.
    x : np.array
        Solution.
    normfk : np.array
        Scaled root mean square residual.
    k : int
        Number of iterations.

    """
    fxk = f(x)
    if scale is None:
        scale = np.ones(x.shape)

    normfk = scaled_norm2(fxk, scale)
    normdx = 0.0

    k = 0
    nNonMonotoneResidual = 0

    if not quiet:
        print('%4i  %12e  %12e  %7f' % (k, normdx, normfk, tau))

    while True:
        if normfk <= tol:
            return (0, x, normfk, k)

        jacMat = jacF(x)
        jacMat = jacrow_scale(jacMat, scale)

        if variant:
            # Solve scaled system
            # (1/tau * I - F'(x)) * dx = F(x), update x <- x + dx
            jacQ, jacR = scipy.linalg.qr(np.diag(1 / (tau * scale)) - jacMat)
        else:
            # Solve scaled system
            # (I - tau * F'(x)) * dx = F(x), update x <- x + tau * dx
            jacQ, jacR = scipy.linalg.qr(np.diag(1 / scale) - tau * jacMat)

        dx = fxk / scale
        try:
            dx = scipy.linalg.solve_triangular(jacR, np.dot(jacQ.T, dx))
        except scipy.linalg.LinAlgError:
            # Singular matrix
            return (-1, x, normfk, k)

        # Update current position
        if variant:
            x = x + dx
        else:
            x = x + tau * dx

        # Calculate residual at new position
        fxk = f(x)
        normfkp1 = scaled_norm2(fxk, scale)

        # Protect against ZeroDivisionError
        if normfkp1 == 0.0:
            return (0, x, normfkp1, k)

        # Update step size
        tau = tau * normfk / normfkp1

        # Check monotonicity of residual
        if normfkp1 < normfk:
            nNonMonotoneResidual = 0
        else:
            nNonMonotoneResidual = nNonMonotoneResidual + 1
            if nNonMonotoneResidual >= maxNonMonotone:
                return (-2, x, normfkp1, k)

        normfk = normfkp1
        k = k + 1

        if not quiet:
            normdx = norm2(dx)
            print('%4i  %12e  %12e  %7f' % (k, normdx, normfk, tau))

        # Check terminal condition
        if k > maxIter:
            break

    return (1, x, normfk, k)
