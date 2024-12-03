from functools import wraps
from typing import Any, NoReturn

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from CADETProcess.dataStructure import Structure, Typed
from CADETProcess.optimization import Population


class SurrogateModel(Structure):
    """
    Surrogate model for an evaluated population.

    Attributes
    ----------
    population : Population
        A population containing evaluated individuals.
    """

    population = Typed(ty=Population)

    def __init__(
            self,
            population: Population,
            *args, **kwargs
            ) -> NoReturn:
        """
        Initialize the Surrogate Model class.

        Parameters
        ----------
        population : Population
            A population containing evaluated individuals.
        """
        super().__init__(*args, population=population, **kwargs)

        self.surrogates: dict[str, dict] = {}
        for eval_fun in [
                'objectives',
                'nonlinear_constraints',
                'nonlinear_constraints_violation',
                'meta_scores',
                ]:
            self.surrogates[eval_fun] = {}
            self.surrogates[eval_fun]['gp']: GaussianProcessRegressor = None
            self.surrogates[eval_fun]['x_scaler']: StandardScaler = None
            self.surrogates[eval_fun]['y_scaler']: StandardScaler = None

        self._update_surrogate_models()

    def train_gp(
            self,
            X: np.ndarray,
            Y: np.ndarray
            ) -> tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]:
        """
        Fit a Gaussian Process on scaled input and output.

        Parameters
        ----------
        X : np.ndarray
            Feature vectors of training data (also required for prediction).
        Y : np.ndarray
            Target values in training data (also required for prediction).

        Returns
        -------
        tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]
            A tuple containing the gaussian process regressor, as well as scalers for
            input and output dimensions.

        """
        X_scaler = StandardScaler().fit(X)
        Y_scaler = StandardScaler().fit(Y)

        gpr = GaussianProcessRegressor()
        gpr.fit(X=X_scaler.transform(X), y=Y_scaler.transform(Y))

        return gpr, X_scaler, Y_scaler

    def _update_eval_fun_surrogate(
            self,
            eval_fun: str,
            surrogate: GaussianProcessRegressor,
            x_scaler: StandardScaler,
            y_scaler: StandardScaler
            ) -> NoReturn:
        self.surrogates[eval_fun]['surrogate'] = surrogate
        self.surrogates[eval_fun]['x_scaler'] = x_scaler
        self.surrogates[eval_fun]['y_scaler'] = y_scaler

    def _evaluate_surrogate(
            self,
            eval_fun: str,
            X: np.ndarray,
            return_std: bool = False,
            return_cov: bool = False,
            ) -> np.ndarray:
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        surrogate = self.surrogates[eval_fun]['surrogate']
        x_scaler = self.surrogates[eval_fun]['x_scaler']
        y_scaler = self.surrogates[eval_fun]['y_scaler']

        X_scaled = x_scaler.transform(X)

        if return_std:
            _, Y_std_scaled = surrogate.predict(
                X_scaled, return_std=True
            )
            Y_std = Y_std_scaled * y_scaler.scale_
            return np.array(Y_std, ndmin=2)
        elif return_cov:
            _, Y_cov_scaled = surrogate.predict(
                X_scaled, return_cov=True
            )
            Y_cov = Y_cov_scaled * (y_scaler.scale_ ** 2)
            return np.array(Y_cov, ndmin=2)
        else:
            Y_scaled = surrogate.predict(X_scaled)
            Y = y_scaler.inverse_transform(np.array(Y_scaled, ndmin=2))
            return Y

    def _update_surrogate_models(self) -> NoReturn:
        if self.population.n_f > 0:
            surrogate, x_scaler, y_scaler = self.train_gp(
                self.population.x, self.population.f
            )
            self._update_eval_fun_surrogate(
                'objectives', surrogate, x_scaler, y_scaler
            )
        if self.population.n_g > 0:
            surrogate, x_scaler, y_scaler = self.train_gp(
                self.population.x, self.population.g
            )
            self._update_eval_fun_surrogate(
                'nonlinear_constraints', surrogate, x_scaler, y_scaler
            )
        if self.population.n_g > 0:
            surrogate, x_scaler, y_scaler = self.train_gp(
                self.population.x, self.population.cv
            )
            self._update_eval_fun_surrogate(
                'nonlinear_constraints_violation', surrogate, x_scaler, y_scaler
            )
        if self.population.n_m > 0:
            surrogate, x_scaler, y_scaler = self.train_gp(
                self.population.x, self.population.m
            )
            self._update_eval_fun_surrogate(
                'meta_scores', surrogate, x_scaler, y_scaler
            )

    def update(self, population: Population) -> NoReturn:
        """
        Update the surrogate model with new population.

        Parameters
        ----------
        population : Population
            New population entries.
        """
        self.population.update(population)
        self._update_surrogate_models()

    def ensures2d(func):
        """Decorate function to ensure X array is an ndarray with ndmin=2."""
        @wraps(func)
        def wrapper(
                self,
                X: np.ndarray,
                *args, **kwargs
                ) -> Any:

            X = np.array(X)
            X_2d = np.array(X, ndmin=2)

            Y = func(self, X_2d, *args, **kwargs)
            Y_2d = Y.reshape((len(X_2d), -1))

            # return an individual or a population depending on the length of X
            if X.ndim == 1:
                return Y_2d[0]
            else:
                return Y_2d

        return wrapper

    @ensures2d
    def estimate_objectives(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the objective function values using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The estimated objective function values.
        """
        return self._evaluate_surrogate('objectives', X)

    @ensures2d
    def estimate_objectives_standard_deviation(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the standard deviation of the objective function values.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The standard deviation of the estimated objective function values.
        """
        return self._evaluate_surrogate('objectives', X, return_std=True)

    @ensures2d
    def estimate_nonlinear_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the nonlinear constraint function values using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The estimated nonlinear constraints function values.
        """
        return self._evaluate_surrogate('nonlinear_constraints', X)

    @ensures2d
    def estimate_nonlinear_constraints_standard_deviation(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Get the standard deviation of the estimated nonlinear constraint function.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The standard deviation of the estimated nonlinear constraint function.
        """
        return self._evaluate_surrogate('nonlinear_constraints', X, return_std=True)

    @ensures2d
    def estimate_nonlinear_constraints_violation(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the nonlinear constraints function violation using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.ndarray
            The estimated nonlinear constraints violation function values.
        """
        return self._evaluate_surrogate('nonlinear_constraints_violation', X)

    @ensures2d
    def estimate_nonlinear_constraints_violation_standard_deviation(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Get the standard deviation of the estimated nonlinear constraint violation function.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The standard deviation of the estimated nonlinear constraint violation function.
        """
        return self._evaluate_surrogate(
            'nonlinear_constraints_violation',
            X,
            return_std=True
        )

    @ensures2d
    def estimate_check_nonlinear_constraints(self, X: np.ndarray) -> np.array:
        """
        Estimate if nonlinear constraints were violated.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.array
            Boolean array indicating if X were valid, based on nonlinear constraint
            violation.
        """
        CV = self.estimate_nonlinear_constraints_violation(X)
        return np.all(CV < 0, axis=1, keepdims=True)

    @ensures2d
    def estimate_meta_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the meta scores using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The estimated meta scores.
        """
        return self._evaluate_surrogate('meta_scores', X)

    @ensures2d
    def estimate_meta_scores_standard_deviation(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Get the standard deviation of the estimated meta scores.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The standard deviation of the estimated meta scores.
        """
        return self._evaluate_surrogate('meta_scores', X, return_std=True)
