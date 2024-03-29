from dataframe import DataFrame
from methods.fit_method import FitMethod
import numpy as np
from scipy import optimize
from scipy.optimize import least_squares as scipy_ls
from scipy.optimize import curve_fit


class WeightedJointLeastSquares(FitMethod):
    def __init__(self, min_power, max_power, mode='h', weight_parameter=0.001, cp_weight=1, dh_weight=1):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.mode = mode
        if mode == 'j':
            self.name = f"Weighted JLS (powers {min_power} ... {max_power}), {mode} mode, {weight_parameter} weight"
        else:
            self.name = f"Weighted JLS (powers {min_power} ... {max_power}), {mode} mode"
        self.min_power = min_power
        self.max_power = max_power
        self.params = np.ones(self.max_power - self.min_power - 1)
        self.weight_parameter = weight_parameter
        self.cp_weght = cp_weight
        self.dh_weight = dh_weight

    def enthalpy(self, parameters, temperature):
        """Enthalpy function from final coefs, H(T) - H(0)"""
        # assert len(parameters) == len(self.params), "Wrong parameters count for h_draw"
        h_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            h_calculated += temperature ** i * parameter
        return h_calculated

    def delta_enthalpy(self, parameters, temperature):
        """Approx aim. Enthalpy function from final coefficients, H(T) - H(T_0)"""
        return self.enthalpy(parameters, temperature) - self.enthalpy(parameters, self.data_frame.reference_temperature)

    def heat_capacity(self, parameters, temperature):
        """Approx aim. Heat Capacity from final coefficients. Cp(T)"""
        cp_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            if i not in [0]:
                cp_calculated += i * temperature ** (i - 1) * parameter
        return cp_calculated

    def stationary_coefficients(self, aux_coefs, t_ref, c_0, c_1):
        """Imitation of non-linearity. Constrained methods require fixing the first two coefficients.
        This function restores normal Cp&dH from auxiliary approx."""
        a_1 = c_1 - \
              sum([i * aux_coefs[i - self.min_power] * t_ref ** (i - 1) for i in range(self.min_power, 0)]) - \
              sum([i * aux_coefs[i - 2 - self.min_power] * t_ref ** (i - 1) for i in range(2, self.max_power + 1)])
        a_0 = c_0 - a_1 * t_ref - \
              sum([aux_coefs[i - self.min_power] * t_ref ** i for i in range(self.min_power, 0)]) - \
              sum([aux_coefs[i - 2 - self.min_power] * t_ref ** i for i in range(2, self.max_power + 1)])

        fit_coefficients = []
        fit_coefficients.extend(aux_coefs[:-self.min_power])
        fit_coefficients.extend([a_0, a_1])
        fit_coefficients.extend(aux_coefs[-self.min_power:])
        return fit_coefficients

    def enthalpy_initial_function(self, parameters, temperature):
        """Approximation of H(T)-H(0), preceding dH calculation.
        This is the constrained part, ignoring powers 0 and 1"""

        assert len(parameters) == len(self.params), "Wrong parameters count for h_draw"

        t_ref = self.data_frame.reference_temperature
        h_calculated = 0.0

        powers = [x for x in range(self.min_power, self.max_power + 1) if x not in [0, 1]]
        for i, parameter in zip(powers, parameters):
            h_calculated += (temperature ** i + (i - 1) * t_ref ** i - i * temperature * t_ref ** (i - 1)) \
                            * parameter
        return h_calculated

    def delta_enthalpy_initial_function(self, temperature, *args):
        """Calling previous function, this one gives _constrained_ H(T)-H(To)"""
        parameters = args
        return self.enthalpy_initial_function(parameters, temperature) - \
               self.enthalpy_initial_function(parameters, self.data_frame.reference_temperature)

    def delta_enthalpy_constrained_cost(self, parameters, temperature, experiment):
        """Returns _constrained_ dH_calc - dH_exp"""
        residuals = (self.delta_enthalpy_initial_function(temperature, *parameters) - experiment) / experiment
        return residuals

    def heat_capacity_initial_function(self, parameters, temperature):
        """Calculates _constrained_ Cp"""
        cp_calculated = 0.0
        t_ref = self.data_frame.reference_temperature
        powers = [x for x in range(self.min_power, self.max_power + 1) if x not in [0, 1]]
        for i, parameter in zip(powers, parameters):
            cp_calculated += i * (temperature ** (i - 1) - t_ref ** (i - 1)) * parameter
        return cp_calculated

    def heat_capacity_cost(self, parameters, temperature, experiment):
        """UNconstrained approximation, takes final coefs and unchanged exp,
        Returns Cp_calc - Cp_exp"""
        residuals = (self.heat_capacity(parameters, temperature) - experiment) / experiment
        return residuals

    def heat_capacity_constrained_cost(self, parameters, temperature, experiment):
        """Constrained Cp approximation
        Requires corrected experiment and shortened coef array"""
        residuals = self.heat_capacity_initial_function(parameters, temperature) - experiment
        return residuals

    def dh_and_cp_calc(self, temperature, *args):
        """United function for joint mode. Both constrained"""
        parameters = args

        dh_temperature = temperature[:len(self.enthalpy_temperature)]
        dh = self.delta_enthalpy_initial_function(dh_temperature, *parameters)

        cp_temperature = temperature[len(self.enthalpy_temperature):]
        cp = self.heat_capacity_initial_function(parameters, cp_temperature)

        return np.concatenate((dh, cp), axis=0)

    def dh_and_cp_relative_error_calc(self, temperature, *args):
        parameters = args

        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = self.data_frame.reference_value
        c_1 = self.data_frame.reference_cvalue

        updated_experiment = self.enthalpy_data - c_0 * np.ones(len(self.enthalpy_temperature)) - c_1 * (
                self.enthalpy_temperature - t_ref_vector)
        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.heat_capacity_temperature))

        dh_temperature = temperature[:len(self.enthalpy_temperature)]
        dh = self.delta_enthalpy_initial_function(dh_temperature, *parameters)
        dh_relative_error = (dh - updated_experiment) / updated_experiment

        cp_temperature = temperature[len(self.enthalpy_temperature):]
        cp = self.heat_capacity_initial_function(parameters, cp_temperature)
        cp_relative_error = (cp - updated_cp) / updated_cp

        print(self.name, sum(dh_relative_error / len(dh_relative_error)) ** 2 + sum(
            cp_relative_error / len(cp_relative_error)) ** 2)

        return np.concatenate((dh_relative_error / len(dh_relative_error), cp_relative_error / len(cp_relative_error)),
                              axis=0)

    def cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh = self.delta_enthalpy_constrained_cost(parameters, h_temp, h_experiment)
        cp = self.heat_capacity_constrained_cost(parameters, cp_temp, cp_experiment)
        return np.concatenate((dh, cp), axis=0)

    def dh_err(self, parameters, temperature, experiment, weights):
        """error function (required for what?)"""
        # todo check if you need it
        residuals = (self.delta_enthalpy_constrained_cost(parameters, temperature, experiment)) ** 2
        print(self.name, parameters, sum(weights * residuals))
        return sum(weights * residuals)

    def cp_err(self, parameters, temperature, experiment, weights):
        """error function (required for what?)"""
        residuals = (self.heat_capacity_cost(parameters, temperature, experiment)) ** 2
        return sum(weights * residuals)

    def joint_err(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment, weights):
        residuals = (self.cost_function(parameters, h_temp, h_experiment, cp_temp, cp_experiment)) ** 2
        return sum(weights * residuals)

    def h_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = self.data_frame.reference_enthalpy_value
        c_1 = self.data_frame.reference_heat_capacity_value

        updated_experiment = self.enthalpy_data - \
                             c_0 * np.ones(len(self.enthalpy_temperature)) - \
                             c_1 * (self.enthalpy_temperature - t_ref_vector)

        self.weights = np.ones(len(self.data_frame.dh_t))

        optimal_params, param_covariation = curve_fit(
            self.delta_enthalpy_initial_function, self.data_frame.dh_t, updated_experiment, self.params,
            sigma=self.weights, absolute_sigma=True)
        print(self.name, optimal_params)

        self.fit_coefficients = self.stationary_coefficients(optimal_params, t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def c_mode_approximation(self):
        self.params = np.ones(self.max_power - self.min_power + 1)
        self.weights = np.ones(len(self.heat_capacity_temperature))
        aux_fit = optimize.minimize(self.cp_err, x0=self.params,
                                    args=(self.data_frame.cp_t, self.data_frame.cp_e, self.weights,))

        self.fit_coefficients = aux_fit.x.tolist()

        # fit_coefficients, соответствующий нулевой степени икса, не влияет на минимум cp, а значит его надо посчитать
        # отдельно, исходя из того что fit(t_ref) = h_ref.
        zero_power_index = -self.min_power
        self.fit_coefficients[zero_power_index] += \
            self.data_frame.reference_enthalpy_value \
            - self.enthalpy(self.fit_coefficients, self.data_frame.reference_temperature)

        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def c_mode_constrained_approximation(self):
        # todo change to appr, add another named unconstrained
        t_ref = self.data_frame.reference_temperature

        c_0 = self.data_frame.reference_enthalpy_value
        c_1 = self.data_frame.reference_heat_capacity_value

        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.heat_capacity_temperature))

        initial_fit = scipy_ls(self.heat_capacity_constrained_cost, self.params,
                               args=(self.data_frame.cp_t, updated_cp))

        self.fit_coefficients = self.stationary_coefficients(initial_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def j_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = self.data_frame.reference_enthalpy_value
        c_1 = self.data_frame.reference_heat_capacity_value

        updated_experiment = self.enthalpy_data - c_0 * np.ones(len(self.enthalpy_temperature)) - c_1 * (
                self.enthalpy_temperature - t_ref_vector)
        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.heat_capacity_temperature))

        self.dh_weights = np.ones(len(self.enthalpy_temperature))
        self.dh_weights *= self.weight_parameter
        self.cp_weights = np.ones(len(self.heat_capacity_temperature))
        self.cp_weights *= (1 - self.weight_parameter)
        self.weights = np.concatenate((self.dh_weights, self.cp_weights), axis=0)

        optimal_params, param_covariation = curve_fit(
            self.dh_and_cp_calc,
            np.concatenate((self.enthalpy_temperature, self.heat_capacity_temperature), axis=0),
            np.concatenate((updated_experiment, updated_cp), axis=0),
            self.params,
            sigma=self.weights, absolute_sigma=False)

        self.fit_coefficients = self.stationary_coefficients(optimal_params, t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def j_relative_error_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = self.data_frame.reference_value
        c_1 = self.data_frame.reference_cvalue

        updated_experiment = self.enthalpy_data - \
                             c_0 * np.ones(len(self.enthalpy_temperature)) - \
                             c_1 * (self.enthalpy_temperature - t_ref_vector)
        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.heat_capacity_temperature))

        self.weights = np.concatenate(
            (0.5 * np.ones(len(self.enthalpy_temperature)), np.ones(len(self.heat_capacity_temperature))), axis=0)

        optimal_params, param_covariation = curve_fit(
            self.dh_and_cp_relative_error_calc,
            np.concatenate((self.enthalpy_temperature, self.heat_capacity_temperature), axis=0),
            np.zeros(len(self.enthalpy_data) + len(self.data_frame.cp_e)),
            self.params,
            sigma=self.weights, absolute_sigma=True)

        print(self.name, optimal_params)

        self.fit_coefficients = self.stationary_coefficients(optimal_params, t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.enthalpy_temperature = data_frame.dh_t
        self.heat_capacity_temperature = data_frame.cp_t
        self.enthalpy_data = data_frame.dh_e

        {
            'h': self.h_mode_approximation,
            'c': self.c_mode_approximation,
            'cc': self.c_mode_constrained_approximation,
            'j': self.j_mode_approximation,
            'j_relative_error': self.j_relative_error_mode_approximation,
        }[self.mode]()

        # print('Finally:', self.name, self.fit_coefficients)

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = \
            (self.data_frame.cp_e - self.heat_capacity(self.fit_coefficients, self.heat_capacity_temperature)) / \
            np.std(self.data_frame.cp_e - self.heat_capacity(self.fit_coefficients, self.heat_capacity_temperature))

    def calculate_enthalpy_residuals(self):
        self.enthalpy_residuals = \
            (self.data_frame.dh_e - self.fit_enthalpy) / np.std(self.data_frame.dh_e - self.fit_enthalpy)

