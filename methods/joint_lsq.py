from dataframe import DataFrame
from methods.fit_method import FitMethod
import numpy as np
from scipy.optimize import least_squares as scipy_ls


class JointLeastSquares(FitMethod):
    def __init__(self, min_power, max_power, mode='h'):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.mode = mode
        self.name = f"Joint LS (powers {min_power} ... {max_power}), {mode} mode"
        self.min_power = min_power
        self.max_power = max_power
        self.params = np.ones(self.max_power - self.min_power - 1)
        # self.params = [1.1, 1.2, 1.3]

    def h_draw(self, parameters, temperature):
        # assert len(parameters) == len(self.params), "Wrong parameters count for h_draw"
        h_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            h_calculated += temperature ** i * parameter
        return h_calculated

    def dh_draw(self, parameters, temperature):
        return self.h_draw(parameters, temperature) - self.h_draw(parameters, self.data_frame.reference_temperature)

    def cp_draw(self, parameters, temperature):
        cp_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            if i not in [0]:
                cp_calculated += i * temperature ** (i - 1) * parameter
        return cp_calculated

    def stationary_coefficients(self, aux_coefs, t_ref, c_0, c_1):
        a_1 = c_1 - \
              sum([i * aux_coefs[i - self.min_power] * t_ref ** (i - 1)
                   for i in range(self.min_power, 0)]) - \
              sum([i * aux_coefs[i - 2 - self.min_power] * t_ref ** (i - 1)
                   for i in range(2, self.max_power + 1)])
        a_0 = c_0 - a_1 * t_ref - sum([aux_coefs[i - self.min_power] * t_ref ** i
                                       for i in range(self.min_power, 0)]) - \
              sum([aux_coefs[i - 2 - self.min_power] * t_ref ** i
                   for i in range(2, self.max_power + 1)])

        fit_coefficients = []
        fit_coefficients.extend(aux_coefs[:-self.min_power])
        fit_coefficients.extend([a_0, a_1])
        fit_coefficients.extend(aux_coefs[-self.min_power:])
        return fit_coefficients

    def h_calc(self, parameters, temperature):
        assert len(parameters) == len(self.params), "Wrong parameters count for h_draw"

        t_ref = self.data_frame.reference_temperature
        h_calculated = 0.0

        powers = [x for x in range(self.min_power, self.max_power + 1) if x not in [0, 1]]
        for i, parameter in zip(powers, parameters):
            h_calculated += (temperature ** i + (i - 1) * t_ref ** i - i * temperature * t_ref ** (i - 1)) \
                            * parameter
        return h_calculated

    def dh_calc(self, parameters, temperature):
        return self.h_calc(parameters, temperature) - self.h_calc(parameters, self.data_frame.reference_temperature)

    def dh_cost(self, parameters, temperature, experiment):
        # residuals = self.dh_calc(parameters, temperature) - experiment
        # renormalize cost
        residuals = (self.dh_calc(parameters, temperature) - experiment) / experiment
        print(self.name, "e_dh:", sum(residuals ** 2))
        return residuals

    def cp_calc(self, parameters, temperature):
        cp_calculated = 0.0
        t_ref = self.data_frame.reference_temperature
        powers = [x for x in range(self.min_power, self.max_power + 1) if x not in [0, 1]]
        for i, parameter in zip(powers, parameters):
            cp_calculated += i * (temperature ** (i - 1) - t_ref ** (i - 1)) * parameter
        return cp_calculated

    def cp_cost(self, parameters, temperature, experiment):
        # residuals = self.cp_draw(parameters, temperature) - experiment
        # renormalize cost
        residuals = (self.cp_draw(parameters, temperature) - experiment) / experiment
        # print(self.name, "e_cp:", sum(residuals ** 2))
        return residuals

    def cp_constrained_cost(self, parameters, temperature, experiment):
        residuals = self.cp_calc(parameters, temperature) - experiment
        # print(self.name, 'e_cp:', sum(residuals ** 2))
        return residuals

    def cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.dh_cost(parameters, h_temp, h_experiment), \
                 self.cp_constrained_cost(parameters, cp_temp, cp_experiment)
        # print(self.name, "e_dh:", sum(dh ** 2), "e_cp:", sum(cp ** 2), "e_dh+cp:",
        #       sum(np.concatenate((dh ** 2, cp ** 2), axis=0)))
        return np.concatenate((dh, cp), axis=0)

    def h_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.temp))

        c_0 = self.data_frame.reference_value
        c_1 = self.data_frame.reference_cvalue

        updated_experiment = self.data_frame.dh_e - c_0 * np.ones(len(self.temp)) - c_1 * (self.temp - t_ref_vector)

        aux_fit = scipy_ls(self.dh_cost, self.params, args=(self.data_frame.dh_t, updated_experiment))
        print(self.name, aux_fit.x.tolist())

        self.fit_coefficients = self.stationary_coefficients(aux_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit = self.dh_draw(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_derivative = self.cp_draw(self.fit_coefficients, self.data_frame.cp_t)

    def c_mode_approximation(self):
        self.params = np.ones(self.max_power - self.min_power + 1)
        aux_fit = scipy_ls(self.cp_cost, self.params, args=(self.data_frame.cp_t, self.data_frame.cp_e))

        self.fit_coefficients = aux_fit.x.tolist()
        self.fit = self.dh_draw(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_derivative = self.cp_draw(self.fit_coefficients, self.data_frame.cp_t)

    def c_mode_constrained_approximation(self):
        t_ref = self.data_frame.reference_temperature

        c_1 = self.data_frame.reference_cvalue

        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.cp_temp))
        # todo AAAAA
        # self.params = scipy_ls(self.cp_cost, [1.0, 1.0, 1.0, 1.0, 1.0], args=(self.data_frame.cp_t, self.data_frame.cp_e)).x.tolist()
        # del self.params[1]
        # del self.params[2]

        aux_fit = scipy_ls(self.cp_constrained_cost, self.params, args=(self.data_frame.cp_t, updated_cp))
        # print(self.name, aux_fit.x.tolist())

        c_0 = self.data_frame.reference_value
        self.fit_coefficients = self.stationary_coefficients(aux_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit = self.dh_draw(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_derivative = self.cp_draw(self.fit_coefficients, self.data_frame.cp_t)

    def j_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.temp))

        c_0 = self.data_frame.reference_value
        c_1 = self.data_frame.reference_cvalue

        updated_experiment = self.experiment - c_0 * np.ones(len(self.temp)) - c_1 * (self.temp - t_ref_vector)
        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.cp_temp))

        aux_fit = scipy_ls(self.cost_function, self.params,
                           args=(self.data_frame.dh_t, updated_experiment, self.data_frame.cp_t, updated_cp))

        self.fit_coefficients = self.stationary_coefficients(aux_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit = self.dh_draw(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_derivative = self.cp_draw(self.fit_coefficients, self.data_frame.cp_t)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.temp = data_frame.dh_t
        self.cp_temp = data_frame.cp_t
        self.experiment = data_frame.dh_e

        {
            'cc': self.c_mode_constrained_approximation,
            'c': self.c_mode_approximation,
            'j': self.j_mode_approximation,
            'h': self.h_mode_approximation
        }[self.mode]()

        print('Finally:', self.name, self.fit_coefficients)

    def calculate_derivative_residuals(self):
        self.derivative_residuals = (self.data_frame.cp_e - self.cp_draw(self.fit_coefficients, self.cp_temp)) / \
                                    np.std(self.data_frame.cp_e - self.cp_draw(self.fit_coefficients, self.cp_temp))


class PlainJointLeastSquares(FitMethod):
    def __init__(self, min_power, max_power):
        self.name = 'Plain Joint LS'
        self.min_power = min_power
        self.max_power = max_power
        self.params = np.ones(self.max_power - self.min_power + 1)

    def h_draw(self, parameters, temperature):
        h_calculated = 0.0
        for i, k in zip(range(self.min_power, self.max_power + 1), range(len(self.params))):
            h_calculated += temperature ** i * parameters[k]
        return h_calculated

    def dh_draw(self, parameters, temperature):
        return self.h_draw(parameters, temperature) - self.h_draw(parameters, self.data_frame.reference_temperature)

    def dh_cost(self, parameters, temperature, experiment):
        return self.h_draw(parameters, temperature) - experiment

    def cp_draw(self, parameters, temperature):
        cp_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            cp_calculated += i * temperature ** (i - 1) * parameter
        return cp_calculated

    def cp_cost(self, parameters, temperature, experiment):
        return self.cp_draw(parameters, temperature) - experiment

    def cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.dh_cost(parameters, h_temp, h_experiment), self.cp_cost(parameters, cp_temp, cp_experiment)
        return np.concatenate((dh, cp), axis=0)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.temp = data_frame.dh_t
        self.cp_temp = data_frame.cp_t
        self.experiment = data_frame.dh_e

        # aux_fit = scipy_ls(self.dh_cost, self.params,  # bounds=self.bounds,
        #                    args=(self.data_frame.dh_t, self.experiment))
        #
        # aux_fit = scipy_ls(self.cost_function, self.params,  # bounds=self.bounds,
        #                    args=(self.data_frame.dh_t, self.data_frame.dh_e, self.data_frame.cp_t, self.data_frame.cp_e))

        # print('init params 2, ', self.params)
        aux_fit = scipy_ls(self.cp_cost, self.params,  # bounds=self.bounds,
                           args=(self.data_frame.cp_t, self.data_frame.cp_e))

        self.fit_coefficients = aux_fit.x.tolist()

        self.fit = self.dh_draw(self.fit_coefficients, self.data_frame.dh_t)

        self.fit_derivative = self.cp_draw(self.fit_coefficients, self.data_frame.dh_t)

    def calculate_residuals(self):
        # if self.mode == 'h':
        self.residuals = (self.data_frame.dh_e - self.fit) / np.std(self.data_frame.dh_e - self.fit)
        # elif self.mode == 'c':
        #     self.residuals = (self.data_frame.cp_e - self.fit_derivative) / \
        #                      np.std(self.data_frame.cp_e - self.fit_derivative)
        # elif self.mode == 'j':
        #    self.residuals = np.concatenate((
        #        (self.data_frame.dh_e - self.fit) / np.std(self.data_frame.dh_e - self.fit),
        #        (self.data_frame.cp_e - self.fit_derivative) / np.std(self.data_frame.cp_e - self.fit_derivative)),
        #        axis=0)

    def calculate_derivative_residuals(self):
        self.derivative_residuals = (self.data_frame.cp_e - self.cp_draw(self.fit_coefficients, self.cp_temp)) / \
                                    np.std(self.data_frame.cp_e - self.cp_draw(self.fit_coefficients, self.cp_temp))
