from dataframe import DataFrame
from methods.auxiliary_function import auxiliary_function, calculate_original_fit
from methods.fit_method import FitMethod
import statsmodels.api as sm
import numpy as np


class WeightedLeastSquaresWithAuxiliaryFunction(FitMethod):
    """Weighted least squares fit for approximation up to given power using auxiliary function."""

    def __init__(self, power):
        self.power = power
        self.name = "WLS Aux, (pow=%d)" % self.power

    def fit(self, data_frame: DataFrame):
        self.temperature = data_frame.dh_t
        self.enthalpy_data = data_frame.dh_e
        self.heat_capacity_data = data_frame.cp_e
        self.data_frame = data_frame

        (self.aux_values, self.aux_weights) = auxiliary_function(data_frame)
        self.aux_source_matrix = np.column_stack([self.temperature ** i for i in range(1, self.power + 1)])
        self.aux_source_matrix = sm.add_constant(self.aux_source_matrix)

        self.aux_fit = sm.WLS(self.aux_values, self.aux_source_matrix, weights=self.aux_weights).fit()

        self.aux_fit_coefficients = self.aux_fit.params

        self.source_matrix = np.vstack(
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in range(-1, 5)]).T

        original_fit_dict = calculate_original_fit(
            {power: self.aux_fit_coefficients[power] for power in range(0, len(self.aux_fit_coefficients))},
            data_frame.reference_temperature,
            data_frame.reference_heat_capacity_value)
        self.original_fit_coefficients = [original_fit_dict[power] if power in original_fit_dict else 0.0 for power in
                                          range(-1, 5)]

        self.fit = np.dot(self.source_matrix, self.original_fit_coefficients)

        self.heat_capacity_matrix = np.vstack([i * data_frame.cp_t ** (i - 1) for i in range(-1, 5)]).T
        self.fit_heat_capacity = np.dot(self.heat_capacity_matrix, self.original_fit_coefficients)

    def calculate_heat_capacity_residuals(self): # todo redo
        heat_capacity_matrix = np.vstack([i * self.data_frame.cp_t ** (i - 1) for i in range(-1, 5)]).T
        heat_capacity = np.dot(heat_capacity_matrix, self.original_fit_coefficients)
        self.heat_capacity_residuals = \
            (self.data_frame.cp_e - heat_capacity)/np.std(self.data_frame.cp_e - heat_capacity)

    def plot_heat_capacity_residuals(self, ax, **kwargs):
        ax.scatter(self.data_frame.cp_t, self.heat_capacity_residuals, **kwargs)

    def calculate_refpoints(self):
        t_ref_array = np.array([self.data_frame.reference_temperature ** i
                                if i != 0 else 1 for i in range(-1, 5)])

        t_ref_diff_array = [-self.data_frame.reference_temperature ** -2]
        t_ref_diff_array.extend([i * self.data_frame.reference_temperature ** (i - 1) for i in range(5)])
        print(np.dot(t_ref_array, self.original_fit_coefficients),
              np.dot(t_ref_diff_array, self.original_fit_coefficients))

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        cp_temp = hc_data.temp
        cp_source_matrix = np.vstack([i * cp_temp ** (i - 1) for i in range(-1, 5)]).T
        fit_cp = np.dot(cp_source_matrix, self.original_fit_coefficients)
        return experiment_cp - fit_cp
