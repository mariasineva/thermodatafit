from methods.fit_method import FitMethod
from methods.auxiliary_function import cost_function_least_abs_dev as lad_cost
from dataframe import DataFrame
from scipy.optimize import minimize
import numpy as np
import statsmodels


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
        fit_coefficients = minimize(lad_cost, self.initial_coefficients,
                                    args=(self.source_matrix, data_frame.experiment))

        self.fit_coefficients = fit_coefficients.x
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.heat_capacity_matrix = np.vstack([i * data_frame.temp ** (i - 1) for i in range(0, self.power + 1)]).T
        self.fit_heat_capacity = np.dot(self.heat_capacity_matrix, self.fit_coefficients)


class LeastSquaresFitNoFree(FitMethod):
    """Least squares fit for no-free-term polynomial."""

    def __init__(self):
        self.name = "No free term"

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.source_matrix = np.vstack(
            [data_frame.temperature ** i - data_frame.reference_temperature ** i for i in [-1, 1, 2]]).T

        self.aux_fit = statsmodels.OLS(data_frame.experiment, self.source_matrix).fit()
        self.fit_coefficients = self.aux_fit.params

        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.heat_capacity_matrix = np.vstack([i * data_frame.temperature ** (i - 1) for i in [-1, 1, 2]]).T
        self.fit_heat_capacity = np.dot(self.heat_capacity_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        # t_ref_array = [1]
        # t_ref_array.extend(np.array([self.data_frame.reference_temperature ** i for i in range(1, self.power + 1)]))
        t_ref_diff_array = [(i + 1) * self.data_frame.reference_temperature ** i for i in [-2, 0, 1]]
        self.fit_coefficients
        print(0, np.dot(t_ref_diff_array, self.fit_coefficients))
