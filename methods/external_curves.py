from methods.fit_method import FitMethod
from dataframe import DataFrame
import statsmodels.api as sm
import numpy as np


class ExternalCurves(FitMethod):
    """Constrained least squares fit for approximation up to given power."""

    def __init__(self, min_power, max_power, min_temp, max_temp, dh_coefficients, source_name):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"
        assert len(dh_coefficients) == max_power + 1 - min_power

        self.min_power = min_power
        self.max_power = max_power

        self.min_temp = min_temp
        self.max_temp = max_temp

        self.name = source_name

        self.fit_coefficients = dh_coefficients

    @staticmethod
    def create_from_dh_params(min_power, max_power, min_temp, max_temp, dh_coefficients, source_name):
        return ExternalCurves(min_power, max_power, min_temp, max_temp, dh_coefficients, source_name)

    @staticmethod
    def create_from_cp_params(min_power, max_power, min_temp, max_temp, cp_coefficients, source_name):
        dh_min_power = min_power + 1
        dh_max_power = max_power + 1
        dh_coefficients = []
        for cp_power, cp_coefficient in zip(range(min_power, max_power + 1), cp_coefficients):
            dh_power = cp_power + 1
            dh_coefficient = 0
            if cp_power == -1:
                assert cp_coefficient == 0
            else:
                dh_coefficient = cp_coefficient / dh_power
            dh_coefficients.append(dh_coefficient)
        return ExternalCurves(dh_min_power, dh_max_power, min_temp, max_temp, dh_coefficients, source_name)

    def enthalpy(self, parameters, temperature):
        '''final enthalpy function, uses all coefficients after calculating a0 and a1'''
        h_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            h_calculated += temperature ** i * parameter
        return h_calculated

    def heat_capacity(self, parameters, temperature):
        cp_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            if i not in [0]:
                cp_calculated += i * temperature ** (i - 1) * parameter
        return cp_calculated

    def delta_enthalpy(self, parameters, temperature):
        return self.enthalpy(parameters, temperature) - self.enthalpy(parameters, self.data_frame.reference_temperature)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.enthalpy_temperature = data_frame.dh_t
        self.heat_capacity_temperature = data_frame.cp_t
        self.enthalpy_data = data_frame.dh_e

        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.enthalpy_temperature)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.heat_capacity_temperature)

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        heat_capacity_temperature = hc_data.temp
        cp_source_matrix = np.vstack(
            [i * heat_capacity_temperature ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        fit_cp = np.dot(cp_source_matrix, self.fit_coefficients)
        return experiment_cp - fit_cp

    def calculate_heat_capacity_residuals(self):
        heat_capacity_matrix = np.vstack(
            [i * self.heat_capacity_temperature ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        heat_capacity = np.dot(heat_capacity_matrix, self.fit_coefficients)
        self.heat_capacity_residuals = (self.data_frame.cp_e - heat_capacity) / np.std(
            self.data_frame.cp_e - heat_capacity)
