from methods.fit_method import FitMethod
from dataframe import DataFrame
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

        self.ols_fit = sm.OLS(self.enthalpy_data, self.source_matrix).fit()

        self.fit_coefficients = self.ols_fit.params
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.heat_capacity_matrix = np.vstack([i * self.temperature ** (i - 1) for i in range(0, self.power + 1)]).T
        self.fit_heat_capacity = np.dot(self.heat_capacity_matrix, self.fit_coefficients)

    def calculate_heat_capacity_residuals(self):
        heat_capacity_matrix = np.vstack(
            [i * self.heat_capacity_temperature ** (i - 1) for i in range(self.power + 1)]).T
        heat_capacity = np.dot(heat_capacity_matrix, self.fit_coefficients)
        self.heat_capacity_residuals = \
            (self.data_frame.cp_e - heat_capacity) / np.std(self.data_frame.cp_e - heat_capacity)

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
        leverage = self.ols_fit.get_influence().hat_matrix_diag
        cooks_distance = self.ols_fit.get_influence().cooks_distance[0]
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]

        for i in leverage_top_3:
            ax.annotate(self.data_frame.temperature[i], xy=(leverage[i], self.residuals[i]))

    def annotate_cooks_distance(self, ax):
        influence = 4 / len(self.data_frame.temperature)
        cooks_distance = self.ols_fit.get_influence().cooks_distance[0]
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
