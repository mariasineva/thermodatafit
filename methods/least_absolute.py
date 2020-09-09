from methods.fit_method import FitMethod
from methods.auxiliary_function import cost_function_lad
from dataframe import DataFrame
from scipy.optimize import minimize
import numpy as np


class LeastAbsoluteFit(FitMethod):
    """Least absolute deviation fit for approximation up to given power."""

    def __init__(self, power, initial_coefficients):
        self.power = power
        self.name = f"Least absolute (pow={self.power})"
        self.initial_coefficients = initial_coefficients

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.source_matrix = np.vstack(
            [data_frame.temp ** i if i != 0 else np.ones(len(data_frame.temp)) for i in range(0, self.power + 1)]).T
        fit_coefficients = minimize(cost_function_lad, self.initial_coefficients,
                                    args=(self.source_matrix, data_frame.experiment))

        self.fit_coefficients = fit_coefficients.x
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack([i * data_frame.temp ** (i - 1) for i in range(0, self.power + 1)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)
