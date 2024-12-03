from scipy.stats import norm, uniform, expon, binom, poisson
from typing import Optional, Union
import numpy as np


class DistributionBase:
    """
    Base class for all distributions.

    Handles common functionality for sampling, mean, and variance using `scipy.stats`
    distributions.
    """

    def __init__(self, dist):
        """
        Initialize the base distribution.

        Parameters
        ----------
        dist : scipy.stats distribution
            A scipy.stats frozen distribution object.
        """
        self._dist = dist

    def sample(
            self,
            size: Optional[Union[int, tuple]] = None
            ) -> Union[float, np.ndarray]:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of samples to generate. Defaults to None for a single sample.

        Returns
        -------
        ndarray or scalar
            Random sample(s) from the distribution.
        """
        return self._dist.rvs(size=size)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """
        return self._dist.mean()

    def var(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            The variance of the distribution.
        """
        return self._dist.var()


class NormalDistribution(DistributionBase):
    """Represents a normal (Gaussian) distribution."""

    def __init__(self, mu: float, sigma: float):
        """
        Initialize the normal distribution.

        Parameters
        ----------
        mu : float
            Mean of the distribution.
        sigma : float
            Standard deviation of the distribution.

        Raises
        ------
        ValueError
            If sigma is negative.
        """
        if sigma < 0:
            raise ValueError("Sigma must be non-negative.")
        super().__init__(norm(loc=mu, scale=sigma))


class UniformDistribution(DistributionBase):
    """Represents a uniform distribution."""

    def __init__(self, lb: float, ub: float):
        """
        Initialize the uniform distribution.

        Parameters
        ----------
        lb : float
            Lower bound of the distribution.
        ub : float
            Upper bound of the distribution.

        Raises
        ------
        ValueError
            If lb is not less than ub.
        """
        if lb >= ub:
            raise ValueError("Lower bound must be less than upper bound.")
        super().__init__(uniform(loc=lb, scale=ub - lb))


class ExponentialDistribution(DistributionBase):
    """Represents an exponential distribution."""

    def __init__(self, lambda_: float):
        """
        Initialize the exponential distribution.

        Parameters
        ----------
        lambda_ : float
            The rate parameter (inverse of mean).

        Raises
        ------
        ValueError
            If lambda_ is non-positive.
        """
        if lambda_ <= 0:
            raise ValueError("Lambda must be positive.")
        super().__init__(expon(scale=1 / lambda_))


class BinomialDistribution(DistributionBase):
    """Represents a binomial distribution."""

    def __init__(self, n: int, p: float):
        """
        Initialize the binomial distribution.

        Parameters
        ----------
        n : int
            Number of trials.
        p : float
            Probability of success in each trial.

        Raises
        ------
        ValueError
            If n is not positive or p is not in [0, 1].
        """
        if n <= 0:
            raise ValueError("Number of trials (n) must be positive.")
        if not (0 <= p <= 1):
            raise ValueError("Probability (p) must be between 0 and 1.")
        super().__init__(binom(n=n, p=p))


class PoissonDistribution(DistributionBase):
    """Represents a Poisson distribution."""

    def __init__(self, lambda_: float):
        """
        Initialize the Poisson distribution.

        Parameters
        ----------
        lambda_ : float
            The rate parameter (mean and variance).

        Raises
        ------
        ValueError
            If lambda_ is non-positive.
        """
        if lambda_ <= 0:
            raise ValueError("Lambda must be positive.")
        super().__init__(poisson(mu=lambda_))
