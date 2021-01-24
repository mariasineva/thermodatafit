from methods.fit_method import FitMethod
from dataframe import DataFrame
from methods.auxiliary_function import auxiliary_function, calculate_original_fit
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.gofplots import ProbPlot


class OrdinaryLeastSquaresSM(FitMethod):
    """Ordinary least squares fit for approximation up to given power. Basic method."""

    def __init__(self, power):
        self.power = power
        self.name = "Ordinary least squares (pow=%d)" % self.power

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.temperature = data_frame.dh_t
        self.enthalpy_data = data_frame.dh_e
        self.heat_capacity_temperature = data_frame.cp_t
        # todo add try

        self.source_matrix = np.column_stack(
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in range(0, self.power + 1)])

        self.aux_fit = sm.OLS(self.enthalpy_data, self.source_matrix).fit()

        self.fit_coefficients = self.aux_fit.params
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.heat_capacity_matrix = np.vstack([i * self.temperature ** (i - 1) for i in range(0, self.power + 1)]).T
        self.fit_heat_capacity = np.dot(self.heat_capacity_matrix, self.fit_coefficients)

    def calculate_heat_capacity_residuals(self):
        heat_capacity_matrix = np.vstack(
            [i * self.heat_capacity_temperature ** (i - 1) for i in range(self.power + 1)]).T
        heat_capacity = np.dot(heat_capacity_matrix, self.fit_coefficients)
        self.heat_capacity_residuals = (self.data_frame.cp_e - heat_capacity) / np.std(self.data_frame.cp_e - heat_capacity)

    def calculate_refpoints(self):
        t_ref_array = [1]
        t_ref_array.extend([self.data_frame.reference_temperature ** i for i in range(1, self.power + 1)])
        t_ref_diff_array = [self.data_frame.reference_temperature ** i for i in range(self.power)]
        diff_coefficients = self.fit_coefficients[1:]
        print(np.dot(t_ref_array, self.fit_coefficients), np.dot(t_ref_diff_array, diff_coefficients))

    def annotate_residuals(self, ax):
        qq = ProbPlot(self.residuals)
        sorted_residuals = np.flip(np.argsort(np.abs(self.residuals)), 0)
        top_3_residuals = sorted_residuals[:3]
        for r, i in enumerate(top_3_residuals):
            ax.annotate(self.temperature[i],
                        xy=(np.sign(self.residuals[i]) * np.flip(qq.theoretical_quantiles, 0)[r], self.residuals[i]))

    def annotate_leverage(self, ax):
        leverage = self.aux_fit.get_influence().hat_matrix_diag
        cooks_distance = self.aux_fit.get_influence().cooks_distance[0]
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]

        for i in leverage_top_3:
            ax.annotate(self.data_frame.temperature[i], xy=(leverage[i], self.residuals[i]))

    def annotate_cooks_distance(self, ax):
        influence = 4 / len(self.data_frame.temperature)
        cooks_distance = self.aux_fit.get_influence().cooks_distance[0]
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]

        for i, distance in enumerate(cooks_distance):
            if distance > influence:
                ax.annotate(self.data_frame.temperature[i], xy=(self.data_frame.temperature[i], distance))

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        heat_capacity_temperature = hc_data.temp
        cp_source_matrix = np.vstack([i * heat_capacity_temperature ** (i - 1) for i in range(0, self.power + 1)]).T
        fit_cp = np.dot(cp_source_matrix, self.fit_coefficients)
        return experiment_cp - fit_cp


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

        self.temperature = data_frame.dh_t
        self.enthalpy_data = data_frame.dh_e
        self.heat_capacity_temperature = data_frame.cp_t

        t_ref = data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.temperature))

        c_0 = data_frame.reference_enthalpy_value
        c_1 = data_frame.reference_heat_capacity_value

        updated_experiment = self.enthalpy_data - c_0 * np.ones(len(self.temperature)) - \
                             c_1 * (self.temperature - t_ref_vector)

        self.updated_matrix = np.column_stack(
            [self.temperature ** i + (i - 1) * t_ref_vector ** i - i * self.temperature * t_ref_vector ** (i - 1) \
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
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in
             range(self.min_power, self.max_power + 1)]).T

        self.fit = np.dot(self.source_matrix, self.fit_coefficients)
        # todo all self fit to self fit enthalpy

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
        self.heat_capacity_residuals = (self.data_frame.cp_e - heat_capacity) / np.std(self.data_frame.cp_e - heat_capacity)