from methods.fit_method import FitMethod
from dataframe import DataFrame
import statsmodels.api as sm
import numpy as np


class СonstrainedLeastSquaresSM(FitMethod):
    """Constrained least squares fit for approximation up to given power."""

    def __init__(self, min_power, max_power):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.min_power = min_power
        self.max_power = max_power

        self.name = f"Constrained LS (powers {min_power} ... {max_power})"

    def fit(self, data_frame: DataFrame):

        self.data_frame = data_frame

        self.enthalpy_temperature = data_frame.dh_t
        self.enthalpy_data = data_frame.dh_e
        self.heat_capacity_temperature = data_frame.cp_t

        t_ref = data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = data_frame.reference_enthalpy_value
        c_1 = data_frame.reference_heat_capacity_value

        updated_experiment = self.enthalpy_data - c_0 * np.ones(len(self.enthalpy_temperature)) - \
                             c_1 * (self.enthalpy_temperature - t_ref_vector)

        self.updated_matrix = np.column_stack(
            [self.enthalpy_temperature ** i + (
                        i - 1) * t_ref_vector ** i - i * self.enthalpy_temperature * t_ref_vector ** (i - 1) \
             for i in range(self.min_power, self.max_power + 1) if i not in [0, 1]])

        self.initial_fit = sm.OLS(updated_experiment, self.updated_matrix).fit()
        self.initial_coefficients = self.initial_fit.params

        a_1 = c_1 - \
              sum([i * self.initial_coefficients[i - self.min_power] * t_ref ** (i - 1) \
                   for i in range(self.min_power, 0)]) - \
              sum([i * self.initial_coefficients[i - 2 - self.min_power] * t_ref ** (i - 1) \
                   for i in range(2, self.max_power + 1)])
        a_0 = c_0 - a_1 * t_ref - \
              sum([self.initial_coefficients[i - self.min_power] * t_ref ** i \
                   for i in range(self.min_power, 0)]) - \
              sum([self.initial_coefficients[i - 2 - self.min_power] * t_ref ** i \
                   for i in range(2, self.max_power + 1)])

        self.fit_coefficients = []
        self.fit_coefficients.extend(self.initial_coefficients[:-self.min_power])
        self.fit_coefficients.extend([a_0, a_1])
        self.fit_coefficients.extend(self.initial_coefficients[-self.min_power:])

        self.source_matrix = np.vstack(
            [self.enthalpy_temperature ** i if i != 0 else np.ones(len(self.enthalpy_temperature)) for i in
             range(self.min_power, self.max_power + 1)]).T

        self.fit_enthalpy = np.dot(self.source_matrix, self.fit_coefficients)

        self.heat_capacity_matrix = np.vstack(
            [i * self.heat_capacity_temperature ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        self.fit_heat_capacity = np.dot(self.heat_capacity_matrix, self.fit_coefficients)

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        heat_capacity_temperature = hc_data.temp
        cp_source_matrix = np.vstack(
            [i * heat_capacity_temperature ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        fit_cp = np.dot(cp_source_matrix, self.fit_coefficients)
        return experiment_cp - fit_cp

    def calculate_refpoints(self):
        t_ref_array = [self.data_frame.reference_temperature ** i if i != 0
                       else 1 for i in range(self.min_power, self.max_power + 1)]

        if self.min_power < 0:
            t_ref_diff_array = [i * self.data_frame.reference_temperature ** (i - 1) for i in
                                range(self.min_power, 0)]
            t_ref_diff_array.extend(
                [i * self.data_frame.reference_temperature ** (i - 1) for i in range(self.max_power + 1)])
        else:
            t_ref_diff_array = [i * self.data_frame.reference_temperature ** (i - 1) for i in
                                range(self.min_power, self.max_power)]

        print(np.dot(t_ref_array, self.fit_coefficients),
              np.dot(t_ref_diff_array, self.fit_coefficients[:len(t_ref_diff_array)]))

    def calculate_heat_capacity_residuals(self):
        heat_capacity_matrix = np.vstack(
            [i * self.heat_capacity_temperature ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        heat_capacity = np.dot(heat_capacity_matrix, self.fit_coefficients)
        self.heat_capacity_residuals = (self.data_frame.cp_e - heat_capacity) / np.std(
            self.data_frame.cp_e - heat_capacity)
